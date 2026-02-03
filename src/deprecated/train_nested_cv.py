import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from data_manager import DataManager, AlzheimerDataset
from models import get_model

# --- CONFIGURATION ---
OUTER_FOLDS = 3   # Paper standard: 5 or 10. Debug: 3.
INNER_FOLDS = 3   # Tuning folds
EPOCHS = 20
BATCH_SIZE = 32
BOOTSTRAP_N = 1000

# Hyperparameter Grid
PARAM_GRID = [
    {'lr': 1e-4, 'dropout': 0.3, 'hidden': 64},
    {'lr': 1e-4, 'dropout': 0.5, 'hidden': 128},
    {'lr': 5e-5, 'dropout': 0.3, 'hidden': 128},
]

def train_model_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        feats = batch['features'].to(device)
        lbls = batch['label'].to(device).float()
        
        optimizer.zero_grad()
        logits = model(feats).squeeze(1)
        loss = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch['features'].to(device)
            lbls = batch['label'].to(device).float()
            logits = model(feats).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            targets.extend(lbls.cpu().numpy())
    
    try:
        return roc_auc_score(targets, preds), np.array(preds), np.array(targets)
    except:
        return 0.5, np.array(preds), np.array(targets)

def bootstrap_ci(y_true, y_pred, n_boot=1000):
    rng = np.random.RandomState(42)
    boot_scores = []
    for _ in range(n_boot):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2: continue
        boot_scores.append(roc_auc_score(y_true[indices], y_pred[indices]))
    
    sorted_scores = np.sort(boot_scores)
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return lower, upper

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Nested CV | Model: {args.model_type} | Task: {args.tasks}")
    if args.classic_subset:
        print(f"   Classic Subset: {args.classic_subset}")
    
    manager = DataManager(args.metadata, target_groups=['CN', 'AD'])
    all_patients = manager.unique_patients
    labels_for_split = manager.get_labels_for_splitting() 

    # --- OUTER LOOP ---
    outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_ix, test_ix) in enumerate(outer_cv.split(all_patients, labels_for_split)):
        print(f"\n=== Outer Fold {fold_idx+1}/{OUTER_FOLDS} ===")
        
        outer_train_pats = all_patients[train_ix]
        outer_test_pats = all_patients[test_ix]
        
        # --- INNER LOOP (Tuning) ---
        print("   Hyperparameter Tuning...")
        inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
        inner_labels = labels_for_split[train_ix]
        
        best_avg_auc = -1
        best_params = PARAM_GRID[0]
        
        for params in PARAM_GRID:
            fold_aucs = []
            for in_tr_ix, in_val_ix in inner_cv.split(outer_train_pats, inner_labels):
                in_train_pats = outer_train_pats[in_tr_ix]
                in_val_pats = outer_train_pats[in_val_ix]
                
                # Load Split Data
                df_tr, df_val = manager.split_patients(in_train_pats, in_val_pats)
                
                if args.model_type == 'classic':
                    (X_tr, y_tr, i_tr, t_tr), (X_val, y_val, i_val, t_val) = \
                        manager.load_classic_features(df_tr, df_val, args.classic_csv, args.tasks, subset=args.classic_subset)
                    input_dim = X_tr.shape[1]
                else:
                    X_tr, y_tr, i_tr, t_tr = manager.load_embeddings(df_tr, args.embedding_dir, args.tasks, args.model_type)
                    X_val, y_val, i_val, t_val = manager.load_embeddings(df_val, args.embedding_dir, args.tasks, args.model_type)
                    input_dim = X_tr.shape[1] if args.model_type == 'roberta' else None

                # Create Loaders
                dl_tr = DataLoader(AlzheimerDataset(X_tr, y_tr, i_tr, t_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                dl_val = DataLoader(AlzheimerDataset(X_val, y_val, i_val, t_val), batch_size=BATCH_SIZE, shuffle=False)
                
                # Train small model
                model = get_model(args.model_type, input_dim, params['hidden'], params['dropout']).to(device)
                optimizer = optim.Adam(model.parameters(), lr=params['lr'])
                criterion = nn.BCEWithLogitsLoss()
                
                # Fast training for tuning (10 epochs)
                curr_best_auc = 0
                for _ in range(10):
                    train_model_epoch(model, dl_tr, optimizer, criterion, device)
                    auc, _, _ = evaluate(model, dl_val, device)
                    if auc > curr_best_auc: curr_best_auc = auc
                
                fold_aucs.append(curr_best_auc)
            
            avg_auc = np.mean(fold_aucs)
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_params = params
        
        print(f"   Best Params: {best_params} (Inner AUC: {best_avg_auc:.4f})")
        
        # --- REFIT & TEST ---
        # Train on ALL outer training data
        df_train_full, df_test = manager.split_patients(outer_train_pats, outer_test_pats)
        
        if args.model_type == 'classic':
            (X_tr, y_tr, i_tr, t_tr), (X_te, y_te, i_te, t_te) = \
                manager.load_classic_features(df_train_full, df_test, args.classic_csv, args.tasks, subset=args.classic_subset)
            input_dim = X_tr.shape[1]
        else:
            X_tr, y_tr, i_tr, t_tr = manager.load_embeddings(df_train_full, args.embedding_dir, args.tasks, args.model_type)
            X_te, y_te, i_te, t_te = manager.load_embeddings(df_test, args.embedding_dir, args.tasks, args.model_type)
            input_dim = X_tr.shape[1] if args.model_type == 'roberta' else None

        dl_train = DataLoader(AlzheimerDataset(X_tr, y_tr, i_tr, t_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        dl_test = DataLoader(AlzheimerDataset(X_te, y_te, i_te, t_te), batch_size=BATCH_SIZE, shuffle=False)
        
        final_model = get_model(args.model_type, input_dim, best_params['hidden'], best_params['dropout']).to(device)
        optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
        criterion = nn.BCEWithLogitsLoss()
        
        best_test_auc = 0
        best_preds = None
        best_targets = None
        
        for ep in range(EPOCHS):
            train_model_epoch(final_model, dl_train, optimizer, criterion, device)
            auc, preds, targets = evaluate(final_model, dl_test, device)
            
            # Save best epoch result
            if auc > best_test_auc:
                best_test_auc = auc
                best_preds = preds
                best_targets = targets
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_ci(best_targets, best_preds, n_boot=BOOTSTRAP_N)
        print(f"   Result: AUC={best_test_auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        results.append({
            'fold': fold_idx,
            'auc': best_test_auc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'best_params': str(best_params)
        })

    # --- SUMMARY ---
    df = pd.DataFrame(results)
    print("\n\n====== 📊 EXPERIMENT RESULTS ======")
    print(f"Model: {args.model_type}")
    print(f"Task:  {args.tasks}")
    if args.classic_subset:
        print(f"Subset: {args.classic_subset}")
    print("-" * 40)
    print(df[['fold', 'auc', 'ci_lower', 'ci_upper']])
    print("-" * 40)
    print(f"Mean AUC: {df['auc'].mean():.4f}")
    print("===================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--classic_csv", type=str, default=None)
    parser.add_argument("--tasks", nargs='+', default=['Fugu'])
    
    # NEW ARGUMENT
    parser.add_argument("--classic_subset", type=str, choices=['audio', 'language'], default=None,
                        help="Filter classic features by type")
    
    args = parser.parse_args()
    main(args)