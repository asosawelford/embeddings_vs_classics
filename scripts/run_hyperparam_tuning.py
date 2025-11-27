import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import subprocess
import os
import re
import itertools

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Benchmark Settings ---
NUM_OUTER_SPLITS = 5
NUM_INNER_FOLDS = 3
TEST_SIZE = 0.2

# --- Data Paths ---
ALL_META_PATHS = [
    'data/train_metadata_lopera.csv',
    'data/validation_metadata_lopera.csv',
    'data/test_metadata_lopera.csv'
]

# --- Hyperparameter Grid for Tuning ---
HYPERPARAM_GRID = {
    "hidden_size": [1024],
    "dropout": [0.6, 0.8]
}

# --- Fixed Training Parameters ---
FIXED_TRAINING_PARAMS = {
    "embedding_path": "/home/aleph/redlat/walm-time-pooled",
    "pitch_csv": "data/test_metadata_lopera.csv",
    "timing_csv": "data/test_metadata_lopera.csv",
    "imputation_means": "data/test_metadata_lopera.csv",
    "scaling_params": "data/test_metadata_lopera.csv",
    "classes": ["CN", "AD"],
    "mode": "embedding",
    "epochs": "100",
    "batch_size": "32",
    "lr": "0.0001",
    "lr_weights": "0.001",
}


# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================

def parse_best_val_uar_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'val_uar' in df.columns:
            return df['val_uar'].max()
    except FileNotFoundError:
        print(f"!!! WARNING: Tuning metrics file not found at {csv_path}")
    return -1

def parse_final_test_uar_from_log(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            match = re.search(r"Test UAR \(Balanced Accuracy\): (\d\.\d+)", content)
            if match:
                return float(match.group(1))
    except FileNotFoundError:
        print(f"!!! WARNING: Final log file not found at {log_file_path}")
    return None


# ==============================================================================
# --- 3. MAIN BENCHMARKING SCRIPT ---
# ==============================================================================

def main():
    print("--- Starting Robust Benchmark with Nested Hyperparameter Tuning ---")

    df_list = [pd.read_csv(p) for p in ALL_META_PATHS if os.path.exists(p)]
    master_df = pd.concat(df_list, ignore_index=True)
    num_samples = len(master_df)
    num_patients = master_df['record_id'].nunique()
    print(f"Master dataset created with {num_samples} total samples from {num_patients} unique patients.")
    
    X = master_df.index
    y = master_df['clinical_diagnosis'].values

    outer_sss = StratifiedShuffleSplit(n_splits=NUM_OUTER_SPLITS, test_size=TEST_SIZE, random_state=42)
    final_benchmark_scores = []
    
    for outer_split_num, (outer_train_idx, outer_test_idx) in enumerate(outer_sss.split(X, y)):
        print(f"\n{'='*20} OUTER SPLIT {outer_split_num + 1}/{NUM_OUTER_SPLITS} {'='*20}")
        
        outer_train_df = master_df.iloc[outer_train_idx]
        outer_test_df = master_df.iloc[outer_test_idx]

        print("--- Starting Inner Tuning Loop to find best params for this split... ---")
        best_params_for_split = None
        best_avg_inner_uar = -1
        param_combinations = [dict(zip(HYPERPARAM_GRID.keys(), v)) for v in itertools.product(*HYPERPARAM_GRID.values())]

        for i, params in enumerate(param_combinations):
            print(f"  Tuning trial {i+1}/{len(param_combinations)} with params: {params}")
            inner_kfold_scores = []
            
            inner_skf = StratifiedKFold(n_splits=NUM_INNER_FOLDS, shuffle=True, random_state=84)
            inner_X = outer_train_df.index
            inner_y = outer_train_df['clinical_diagnosis'].values
            
            for inner_fold_num, (inner_train_idx, inner_val_idx) in enumerate(inner_skf.split(inner_X, inner_y)):
                inner_train_df = outer_train_df.iloc[inner_train_idx]
                inner_val_df = outer_train_df.iloc[inner_val_idx]
                
                temp_train_path = 'data/temp_inner_train.csv'
                temp_val_path = 'data/temp_inner_val.csv'
                inner_train_df.to_csv(temp_train_path, index=False)
                inner_val_df.to_csv(temp_val_path, index=False)
                
                results_dir = f"results/tuning/outer_{outer_split_num+1}/trial_{i+1}/fold_{inner_fold_num+1}"
                
                command = [
                    "python", "src/train.py",
                    "--train_meta", temp_train_path, "--val_meta", temp_val_path, "--test_meta", temp_val_path,
                    "--patience", "15", "--hidden_size", str(params["hidden_size"]), "--dropout", str(params["dropout"]),
                    "--results_dir", results_dir, "--plots_dir", "plots/tuning_plots",
                    "--save_path", os.path.join(results_dir, "best_model.pth")
                ]
                
                for key, val in FIXED_TRAINING_PARAMS.items():
                    if isinstance(val, list): command.extend([f"--{key}", *val])
                    else: command.extend([f"--{key}", val])
                
                command.extend(["--embedding_layers", *[str(l) for l in range(13)]])

                # --- FIX #1: ADD ROBUST ERROR REPORTING ---
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"!!! ERROR during tuning run for params {params} !!!")
                    print(f"Return Code: {e.returncode}")
                    print("--- STDOUT ---")
                    print(e.stdout)
                    print("--- STDERR ---")
                    print(e.stderr)
                    raise # Stop the script
                # --- END FIX #1 ---

                experiment_name = f"{FIXED_TRAINING_PARAMS['classes'][0]}_vs_{FIXED_TRAINING_PARAMS['classes'][1]}_{FIXED_TRAINING_PARAMS['mode']}_layers_{'_'.join(map(str, range(13)))}"
                metrics_csv = os.path.join(results_dir, f'{experiment_name}_epoch_metrics.csv')
                best_val_uar = parse_best_val_uar_from_csv(metrics_csv)
                inner_kfold_scores.append(best_val_uar)

            avg_uar_for_params = np.mean(inner_kfold_scores)
            print(f"    -> Average validation UAR for these params: {avg_uar_for_params:.4f}")

            if avg_uar_for_params > best_avg_inner_uar:
                best_avg_inner_uar = avg_uar_for_params
                best_params_for_split = params
        
        print(f"\n--- Best Hyperparameters for Outer Split {outer_split_num + 1}: {best_params_for_split} (UAR: {best_avg_inner_uar:.4f}) ---")

        print("--- Training final model on full outer_train set and evaluating on outer_test set... ---")
        temp_train_path = 'data/temp_outer_train.csv'
        temp_test_path = 'data/temp_outer_test.csv'
        outer_train_df.to_csv(temp_train_path, index=False)
        outer_test_df.to_csv(temp_test_path, index=False)

        results_dir = f"results/final_benchmark/split_{outer_split_num + 1}"
        plots_dir = f"plots/final_benchmark/split_{outer_split_num + 1}"

        final_command = [
            "python", "src/train.py",
            "--train_meta", temp_train_path, "--val_meta", temp_test_path, "--test_meta", temp_test_path,
            "--patience", "20", "--hidden_size", str(best_params_for_split["hidden_size"]), "--dropout", str(best_params_for_split["dropout"]),
            "--results_dir", results_dir, "--plots_dir", plots_dir,
            "--save_path", os.path.join(results_dir, "best_model.pth")
        ]
        
        # --- FIX #2: CORRECTLY EXTEND 'final_command' ---
        for key, val in FIXED_TRAINING_PARAMS.items():
            if isinstance(val, list):
                final_command.extend([f"--{key}", *val])
            else:
                final_command.extend([f"--{key}", val])
        # --- END FIX #2 ---

        final_command.extend(["--embedding_layers", *[str(l) for l in range(13)]])

        try:
            subprocess.run(final_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR during final evaluation run for split {outer_split_num + 1} !!!")
            print(f"Return Code: {e.returncode}")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            raise

        experiment_name = f"{FIXED_TRAINING_PARAMS['classes'][0]}_vs_{FIXED_TRAINING_PARAMS['classes'][1]}_{FIXED_TRAINING_PARAMS['mode']}_layers_{'_'.join(map(str, range(13)))}"
        log_file = os.path.join(results_dir, f'{experiment_name}_results.txt')
        final_test_uar = parse_final_test_uar_from_log(log_file)
        
        if final_test_uar is not None:
            print(f"--- RESULT FOR OUTER SPLIT {outer_split_num + 1}: Test UAR = {final_test_uar:.4f} ---")
            final_benchmark_scores.append(final_test_uar)

    print("\n\n{'='*20} BENCHMARK COMPLETE {'='*20}")
    if final_benchmark_scores:
        mean_uar = np.mean(final_benchmark_scores)
        std_uar = np.std(final_benchmark_scores)
        print(f"Final UAR Scores across {NUM_OUTER_SPLITS} outer splits: {[f'{s:.4f}' for s in final_benchmark_scores]}")
        print(f"\n--> Final Robust Benchmark UAR: {mean_uar:.4f} ± {std_uar:.4f}")
    else:
        print("No final scores were collected. Please check for errors in the final evaluation runs.")

if __name__ == "__main__":
    main()