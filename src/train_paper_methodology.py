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
MAX_EPOCHS = 50     
PATIENCE = 15        
BATCH_SIZE = 32


# Hyperparameter Space
SPACE = [
    Real(1e-5, 1e-3, name='lr', prior='log-uniform'),
    Real(0.2, 0.6, name='dropout'),
    Categorical([128, 256, 512], name='hidden')
]

# --- HELPER: Create Differential Optimizer ---
def get_optimizer(model, base_lr):
    """
    Sets base_lr for the model, but forces a higher LR (0.01) 
    for the WavLM WeightedAverageLayer weights.
    """
    agg_params = []
    rest_params = []
    
    for name, param in model.named_parameters():
        # Identify the learnable weights in aggregator
        if ('aggregator.weights' in name) or ('wavlm_agg.weights' in name):
            agg_params.append(param)
        else:
            rest_params.append(param)
            
    if len(agg_params) > 0:
        # Differential Learning Rates
        return optim.Adam([
            {'params': rest_params, 'lr': base_lr},
            {'params': agg_params, 'lr': 0.01} # Force higher LR for weights
        ])
    else:
        # Standard Optimizer (for Classic/RoBERTa)
        return optim.Adam(model.parameters(), lr=base_lr)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif self.mode == 'max' and current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'min' and current_score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

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
        
        if 'features_text' in batch:
            feats_text = batch['features_text'].to(device)
            logits = model(feats, feats_text).squeeze(1)
        else:
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
            
            if 'features_text' in batch:
                feats_text = batch['features_text'].to(device)
                logits = model(feats, feats_text).squeeze(1)
            else:
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
                    
                    if args.model_type == 'fusion':
                        X_w, X_r, y = manager.load_paired_embeddings(df_tr, args.embedding_dir, args.roberta_dir, args.tasks)
                        ds_tr = AlzheimerDataset(X_w, y, features_text=X_r)
                        X_w_v, X_r_v, y_v = manager.load_paired_embeddings(df_val, args.embedding_dir, args.roberta_dir, args.tasks)
                        ds_val = AlzheimerDataset(X_w_v, y_v, features_text=X_r_v)
                        input_dim = None
                    elif args.model_type == 'classic':
                        (X_tr, y_tr, _, _), (X_val, y_val, _, _) = manager.load_classic_features(df_tr, df_val, args.classic_csv, args.tasks, subset=args.classic_subset)
                        ds_tr = AlzheimerDataset(X_tr, y_tr)
                        ds_val = AlzheimerDataset(X_val, y_val)
                        input_dim = X_tr.shape[1]
                    else:
                        X_tr, y_tr, _, _ = manager.load_embeddings(df_tr, args.embedding_dir, args.tasks, args.model_type)
                        X_val, y_val, _, _ = manager.load_embeddings(df_val, args.embedding_dir, args.tasks, args.model_type)
                        ds_tr = AlzheimerDataset(X_tr, y_tr)
                        ds_val = AlzheimerDataset(X_val, y_val)
                        input_dim = X_tr.shape[1] if args.model_type == 'roberta' else None

                    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
                    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
                    
                    model = get_model(args.model_type, input_dim, params['hidden'], params['dropout']).to(device)
                    # USE HELPER FOR DIFFERENTIAL LR
                    optimizer = get_optimizer(model, params['lr'])
                    criterion = nn.BCEWithLogitsLoss()
                    
                    # --- INNER EARLY STOPPING ---
                    stopper = EarlyStopping(patience=PATIENCE, mode='max')
                    for _ in range(MAX_EPOCHS):
                        train_model_epoch(model, dl_tr, optimizer, criterion, device)
                        auc, _, _ = evaluate(model, dl_val, device)
                        stopper(auc, model)
                        if stopper.early_stop: break
                    inner_aucs.append(stopper.best_score)
                
                return -np.mean(inner_aucs)

            res_gp = gp_minimize(objective, SPACE, n_calls=BAYES_ITER, n_initial_points=BAYES_INIT, random_state=seed, verbose=False)
            best_params = {'lr': float(res_gp.x[0]), 'dropout': float(res_gp.x[1]), 'hidden': int(res_gp.x[2])}
            
            # --- REFIT & TEST ---
            df_train_full, df_test = manager.split_patients(outer_train_pats, outer_test_pats)
            
            if args.model_type == 'fusion':
                X_w, X_r, y = manager.load_paired_embeddings(df_train_full, args.embedding_dir, args.roberta_dir, args.tasks)
                ds_train = AlzheimerDataset(X_w, y, features_text=X_r)
                X_w_t, X_r_t, y_t = manager.load_paired_embeddings(df_test, args.embedding_dir, args.roberta_dir, args.tasks)
                ds_test = AlzheimerDataset(X_w_t, y_t, features_text=X_r_t)
                input_dim = None
            elif args.model_type == 'classic':
                (X_tr, y_tr, _, _), (X_te, y_te, _, _) = manager.load_classic_features(df_train_full, df_test, args.classic_csv, args.tasks, subset=args.classic_subset)
                ds_train = AlzheimerDataset(X_tr, y_tr)
                ds_test = AlzheimerDataset(X_te, y_te)
                input_dim = X_tr.shape[1]
            else:
                X_tr, y_tr, _, _ = manager.load_embeddings(df_train_full, args.embedding_dir, args.tasks, args.model_type)
                X_te, y_te, _, _ = manager.load_embeddings(df_test, args.embedding_dir, args.tasks, args.model_type)
                ds_train = AlzheimerDataset(X_tr, y_tr)
                ds_test = AlzheimerDataset(X_te, y_te)
                input_dim = X_tr.shape[1] if args.model_type == 'roberta' else None

            dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
            
            final_model = get_model(args.model_type, input_dim, best_params['hidden'], best_params['dropout']).to(device)
            # USE HELPER FOR DIFFERENTIAL LR
            optimizer = get_optimizer(final_model, best_params['lr'])
            criterion = nn.BCEWithLogitsLoss()
            
            # --- OUTER EARLY STOPPING ---
            final_stopper = EarlyStopping(patience=PATIENCE, mode='max') # <--- UPDATED
            for ep in range(MAX_EPOCHS): # <--- UPDATED
                train_model_epoch(final_model, dl_train, optimizer, criterion, device)
                auc, _, _ = evaluate(final_model, dl_test, device)
                final_stopper(auc, final_model) # <--- UPDATED
                if final_stopper.early_stop: break # <--- UPDATED
            
            final_model.load_state_dict(final_stopper.best_weights) # <--- UPDATED: Restore best model
            best_test_auc, best_preds, best_targets = evaluate(final_model, dl_test, device) # <--- UPDATED
            
            ci_lower, ci_upper = bootstrap_ci(best_targets, best_preds, n_boot=BOOTSTRAP_N)
            
            res_dict = {
                'seed': seed, 'fold': fold_idx, 'auc': best_test_auc,
                'ci_lower': ci_lower, 'ci_upper': ci_upper,
                'lr': best_params['lr'], 'dropout': best_params['dropout'], 'hidden': best_params['hidden']
            }

            # --- EXTRACT LAYER WEIGHTS ---
            if hasattr(final_model, 'get_layer_weights'):
                weights = final_model.get_layer_weights()
                for i, w in enumerate(weights):
                    res_dict[f'L{i}'] = float(w)
            
            final_results.append(res_dict)
            pd.DataFrame(final_results).to_csv(save_dir / "results_partial.csv", index=False)

    df = pd.DataFrame(final_results)
    df.to_csv(save_dir / "results.csv", index=False)
    
    logging.info("\n\n====== 📜 FINAL MANUSCRIPT RESULTS ======")
    logging.info(df)
    logging.info("-" * 50)
    logging.info(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--roberta_dir", type=str, default=None)
    parser.add_argument("--classic_csv", type=str, default=None)
    parser.add_argument("--tasks", nargs='+', default=['Fugu'])
    parser.add_argument("--classic_subset", type=str, choices=['audio', 'language', 'combined'], default=None)
    parser.add_argument("--quick_debug", action="store_true")
    parser.add_argument("--target_groups", nargs='+', default=['CN', 'AD'])
    
    args = parser.parse_args()
    if args.quick_debug:
        print("⚠️ DEBUG MODE")
        N_SEEDS = 1; OUTER_FOLDS = 2; INNER_FOLDS = 2; BAYES_INIT = 2; BAYES_ITER = 2
    main(args)