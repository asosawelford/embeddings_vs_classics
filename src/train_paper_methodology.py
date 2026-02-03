import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
import copy
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from data_manager import DataManager, AlzheimerDataset
from models import get_model

# # --- VERY RIGOROUS CONFIGURATION ---
# N_SEEDS = 5
# OUTER_FOLDS = 3
# INNER_FOLDS = 11  # As per your text
# BAYES_INIT = 15   # Initial random points
# BAYES_ITER = 50   # Optimization iterations
# BOOTSTRAP_N = 1000
# EPOCHS = 20
# BATCH_SIZE = 32

# --- mas chill
N_SEEDS = 3
OUTER_FOLDS = 2
INNER_FOLDS = 5  # As per your text
BAYES_INIT = 5   # Initial random points
BAYES_ITER = 10  # Optimization iterations
BOOTSTRAP_N = 1000
EPOCHS = 20
BATCH_SIZE = 32


# Hyperparameter Space
SPACE = [
    Real(1e-5, 1e-3, name='lr', prior='log-uniform'),
    Real(0.2, 0.6, name='dropout'),
    Categorical([128, 256, 512], name='hidden')
]

# --- LOGGING SETUP ---
def setup_experiment(args):
    """Creates output folder and sets up logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a descriptive name: 2023..._FTD_vs_CN_classic_language
    group_str = "_vs_".join(args.target_groups)
    subset_str = f"_{args.classic_subset}" if args.classic_subset else ""
    
    exp_name = f"{timestamp}_{group_str}_{args.model_type}{subset_str}"
    
    save_dir = Path("experiments") / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logging setup
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(save_dir / "run.log"),
            logging.StreamHandler()
        ]
    )

    # Save Config
    config = vars(args)
    config['CONSTANTS'] = {
        'N_SEEDS': N_SEEDS,
        'OUTER_FOLDS': OUTER_FOLDS,
        'INNER_FOLDS': INNER_FOLDS,
        'BAYES_INIT': BAYES_INIT,
        'BAYES_ITER': BAYES_ITER
    }
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    logging.info(f"📂 Experiment Output Directory: {save_dir}")
    return save_dir

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
    save_dir = setup_experiment(args)
    
    logging.info(f"🚀 Starting Paper Methodology Run")
    logging.info(f"   Target Groups: {args.target_groups}") # Log the specific comparison
    logging.info(f"   Model: {args.model_type} | Task: {args.tasks}")
    logging.info(f"   Config: {N_SEEDS} Seeds | {OUTER_FOLDS}x{INNER_FOLDS} Nested CV | BayesOpt ({BAYES_INIT} init, {BAYES_ITER} iter)")

    # --- UPDATE: Pass the command line target_groups to DataManager ---
    manager = DataManager(args.metadata, target_groups=args.target_groups)
    
    all_patients = manager.unique_patients
    labels_for_split = manager.get_labels_for_splitting() 

    final_results = []

    for seed in range(N_SEEDS):
        logging.info(f"\n🌱 === SEED {seed+1}/{N_SEEDS} ===")
        outer_cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        
        for fold_idx, (train_ix, test_ix) in enumerate(outer_cv.split(all_patients, labels_for_split)):
            logging.info(f"   Outer Fold {fold_idx+1}/{OUTER_FOLDS}")
            
            outer_train_pats = all_patients[train_ix]
            outer_test_pats = all_patients[test_ix]
            
            # --- BAYESIAN OPTIMIZATION ---
            @use_named_args(SPACE)
            def objective(**params):
                inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
                inner_labels = labels_for_split[train_ix]
                inner_aucs = []
                
                for in_tr_ix, in_val_ix in inner_cv.split(outer_train_pats, inner_labels):
                    in_train_pats = outer_train_pats[in_tr_ix]
                    in_val_pats = outer_train_pats[in_val_ix]
                    
                    df_tr, df_val = manager.split_patients(in_train_pats, in_val_pats)
                    
                    if args.model_type == 'classic':
                        (X_tr, y_tr, i_tr, t_tr), (X_val, y_val, i_val, t_val) = \
                            manager.load_classic_features(df_tr, df_val, args.classic_csv, args.tasks, subset=args.classic_subset)
                        input_dim = X_tr.shape[1]
                    else:
                        X_tr, y_tr, i_tr, t_tr = manager.load_embeddings(df_tr, args.embedding_dir, args.tasks, args.model_type)
                        X_val, y_val, i_val, t_val = manager.load_embeddings(df_val, args.embedding_dir, args.tasks, args.model_type)
                        input_dim = X_tr.shape[1] if args.model_type == 'roberta' else None

                    dl_tr = DataLoader(AlzheimerDataset(X_tr, y_tr, i_tr, t_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                    dl_val = DataLoader(AlzheimerDataset(X_val, y_val, i_val, t_val), batch_size=BATCH_SIZE, shuffle=False)
                    
                    model = get_model(args.model_type, input_dim, params['hidden'], params['dropout']).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
                    criterion = nn.BCEWithLogitsLoss()
                    
                    best_inner_auc = 0
                    for _ in range(10): 
                        train_model_epoch(model, dl_tr, optimizer, criterion, device)
                        auc, _, _ = evaluate(model, dl_val, device)
                        if auc > best_inner_auc: best_inner_auc = auc
                    inner_aucs.append(best_inner_auc)
                
                return -np.mean(inner_aucs)

            res_gp = gp_minimize(objective, SPACE, n_calls=BAYES_ITER, n_initial_points=BAYES_INIT, random_state=seed, verbose=False)
            
            best_params = {
                'lr': float(res_gp.x[0]),
                'dropout': float(res_gp.x[1]),
                'hidden': int(res_gp.x[2])
            }
            
            # --- REFIT & TEST ---
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
                if auc > best_test_auc:
                    best_test_auc = auc
                    best_preds = preds
                    best_targets = targets
            
            ci_lower, ci_upper = bootstrap_ci(best_targets, best_preds, n_boot=BOOTSTRAP_N)
            
            res_dict = {
                'seed': seed,
                'fold': fold_idx,
                'auc': best_test_auc,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'lr': best_params['lr'],
                'dropout': best_params['dropout'],
                'hidden': best_params['hidden']
            }
            final_results.append(res_dict)
            pd.DataFrame(final_results).to_csv(save_dir / "results_partial.csv", index=False)

    # --- FINAL REPORT ---
    df = pd.DataFrame(final_results)
    df.to_csv(save_dir / "results.csv", index=False)
    
    logging.info("\n\n====== 📜 FINAL MANUSCRIPT RESULTS ======")
    logging.info(f"Groups: {args.target_groups}") # Print groups in final report
    logging.info(df)
    
    mean_auc = df['auc'].mean()
    std_auc = df.groupby('seed')['auc'].mean().std()
    
    logging.info("-" * 50)
    logging.info(f"Overall Mean AUC: {mean_auc:.4f}")
    logging.info(f"Robustness (Std across seeds): {std_auc:.4f}")
    logging.info(f"Results saved to: {save_dir}")
    logging.info("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--classic_csv", type=str, default=None)
    parser.add_argument("--tasks", nargs='+', default=['Fugu'])
    parser.add_argument("--classic_subset", type=str, choices=['audio', 'language', 'combined'], default=None)
    parser.add_argument("--quick_debug", action="store_true")
    
    # --- NEW ARGUMENT FOR FTD ---
    parser.add_argument("--target_groups", nargs='+', default=['CN', 'AD'], 
                        help="The two groups to classify. e.g. CN FTD")
    
    args = parser.parse_args()
    
    if args.quick_debug:
        print("⚠️ DEBUG MODE")
        N_SEEDS = 1
        OUTER_FOLDS = 2
        INNER_FOLDS = 2
        BAYES_INIT = 2
        BAYES_ITER = 2
        
    main(args)