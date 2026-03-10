import numpy as np
import pandas as pd
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

from data_manager import DataManager

# --- CONFIGURATION ---
N_SEEDS = 1        # Keep consistent with your main run
OUTER_FOLDS = 2
INNER_FOLDS = 5    
BAYES_INIT = 10
BAYES_ITER = 20
BOOTSTRAP_N = 1000

# Hyperparameter Space for SVM
SPACE = [
    Real(1e-3, 100, name='C', prior='log-uniform'),
    Categorical(['linear', 'rbf', 'poly'], name='kernel'),
    Real(1e-4, 1, name='gamma', prior='log-uniform') # Only used for rbf/poly
]

def setup_experiment(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    group_str = "_vs_".join(args.target_groups)
    subset_str = f"_{args.classic_subset}" if args.classic_subset else ""
    exp_name = f"{timestamp}_{group_str}_SVM_{args.model_type}{subset_str}"
    
    save_dir = Path("experiments") / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

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
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    logging.info(f"📂 SVM Experiment: {save_dir}")
    return save_dir

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

def prepare_data_for_svm(X_raw, model_type):
    """
    SVMs need 2D arrays (Samples, Features).
    We must flatten/average any 3D data.
    """
    # If Classic (tuple return from manager), just take X
    if isinstance(X_raw, tuple):
        # Classic usually returns (X_a, X_l, y) or (X, y) depending on function
        # But DataManager returns (X, y, ...) in load_classic
        # Let's handle the numpy array directly
        pass 

    X = X_raw
    
    # WavLM: (N, 13, 768) -> Average -> (N, 768)
    if X.ndim == 3:
        X = np.mean(X, axis=1)
        
    return X

def main(args):
    save_dir = setup_experiment(args)
    logging.info(f"🚀 Starting SVM Run | Task: {args.tasks}")

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
                    
                    # --- DATA LOADING (Reusing your logic) ---
                    if args.model_type == 'fusion' or args.model_type == 'classic_fusion':
                        # For SVM, Fusion = Concatenation
                        if args.model_type == 'fusion':
                            X_a, X_b, y_tr = manager.load_paired_embeddings(df_tr, args.embedding_dir, args.roberta_dir, args.tasks)
                            X_av, X_bv, y_val = manager.load_paired_embeddings(df_val, args.embedding_dir, args.roberta_dir, args.tasks)
                            # WavLM Average
                            if X_a.ndim == 3: X_a = np.mean(X_a, axis=1)
                            if X_av.ndim == 3: X_av = np.mean(X_av, axis=1)
                        else:
                            (X_a, X_b, y_tr), (X_av, X_bv, y_val) = manager.load_dual_classic_features(df_tr, df_val, args.classic_csv, args.tasks)
                        
                        # Concatenate
                        X_tr = np.hstack([X_a, X_b])
                        X_val = np.hstack([X_av, X_bv])

                    elif args.model_type == 'classic':
                        (X_tr, y_tr, _, _), (X_val, y_val, _, _) = manager.load_classic_features(df_tr, df_val, args.classic_csv, args.tasks, subset=args.classic_subset)
                    else: # WavLM or RoBERTa
                        X_tr, y_tr, _, _ = manager.load_embeddings(df_tr, args.embedding_dir, args.tasks, args.model_type)
                        X_val, y_val, _, _ = manager.load_embeddings(df_val, args.embedding_dir, args.tasks, args.model_type)
                        # Average WavLM if needed
                        if X_tr.ndim == 3: X_tr = np.mean(X_tr, axis=1)
                        if X_val.ndim == 3: X_val = np.mean(X_val, axis=1)

                    # --- TRAIN SVM ---
                    # We use a Pipeline to ensure scaling happens (crucial for SVMs)
                    clf = make_pipeline(StandardScaler(), SVC(probability=True, **params, random_state=seed))
                    clf.fit(X_tr, y_tr)
                    
                    # Predict
                    probs = clf.predict_proba(X_val)[:, 1]
                    try:
                        score = roc_auc_score(y_val, probs)
                    except:
                        score = 0.5
                    inner_aucs.append(score)
                
                return -np.mean(inner_aucs)

            # Run Optimization
            res_gp = gp_minimize(objective, SPACE, n_calls=BAYES_ITER, n_initial_points=BAYES_INIT, random_state=seed, verbose=False)
            best_params = {'C': res_gp.x[0], 'kernel': res_gp.x[1], 'gamma': res_gp.x[2]}
            
            # --- REFIT ---
            df_train_full, df_test = manager.split_patients(outer_train_pats, outer_test_pats)
            
            # Load Data (Copy logic from above)
            if args.model_type == 'fusion' or args.model_type == 'classic_fusion':
                if args.model_type == 'fusion':
                    X_a, X_b, y_train = manager.load_paired_embeddings(df_train_full, args.embedding_dir, args.roberta_dir, args.tasks)
                    X_at, X_bt, y_test = manager.load_paired_embeddings(df_test, args.embedding_dir, args.roberta_dir, args.tasks)
                    if X_a.ndim == 3: X_a = np.mean(X_a, axis=1)
                    if X_at.ndim == 3: X_at = np.mean(X_at, axis=1)
                else:
                    (X_a, X_b, y_train), (X_at, X_bt, y_test) = manager.load_dual_classic_features(df_train_full, df_test, args.classic_csv, args.tasks)
                X_train = np.hstack([X_a, X_b])
                X_test_final = np.hstack([X_at, X_bt])
            
            elif args.model_type == 'classic':
                (X_train, y_train, _, _), (X_test_final, y_test, _, _) = manager.load_classic_features(df_train_full, df_test, args.classic_csv, args.tasks, subset=args.classic_subset)
            else:
                X_train, y_train, _, _ = manager.load_embeddings(df_train_full, args.embedding_dir, args.tasks, args.model_type)
                X_test_final, y_test, _, _ = manager.load_embeddings(df_test, args.embedding_dir, args.tasks, args.model_type)
                if X_train.ndim == 3: X_train = np.mean(X_train, axis=1)
                if X_test_final.ndim == 3: X_test_final = np.mean(X_test_final, axis=1)

            # Final Train
            final_clf = make_pipeline(StandardScaler(), SVC(probability=True, **best_params, random_state=seed))
            final_clf.fit(X_train, y_train)
            
            probs = final_clf.predict_proba(X_test_final)[:, 1]
            auc = roc_auc_score(y_test, probs)
            ci_lower, ci_upper = bootstrap_ci(y_test, probs, n_boot=BOOTSTRAP_N)
            
            res_dict = {
                'seed': seed, 'fold': fold_idx, 'auc': auc,
                'ci_lower': ci_lower, 'ci_upper': ci_upper,
                **best_params
            }
            final_results.append(res_dict)
            pd.DataFrame(final_results).to_csv(save_dir / "results_partial.csv", index=False)

    df = pd.DataFrame(final_results)
    df.to_csv(save_dir / "results.csv", index=False)
    logging.info(f"\n\n====== 📜 FINAL SVM RESULTS ======\n{df}\nSaved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--roberta_dir", type=str, default=None)
    parser.add_argument("--classic_csv", type=str, default=None)
    parser.add_argument("--tasks", nargs='+', default=['Fugu'])
    parser.add_argument("--classic_subset", type=str, choices=['audio', 'language', 'combined'], default=None)
    parser.add_argument("--target_groups", nargs='+', default=['CN', 'AD'])
    
    args = parser.parse_args()
    main(args)