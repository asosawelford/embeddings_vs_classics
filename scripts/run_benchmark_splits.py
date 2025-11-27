import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
import os
import re
import argparse

# ==============================================================================
# --- 1. EXPERIMENT CONFIGURATIONS ---
# ==============================================================================

# --- Common Settings for All Benchmarks ---
NUM_SPLITS = 5
TEST_SIZE = 0.2
ALL_META_PATHS = [
    'data/train_metadata_lopera.csv',
    'data/validation_metadata_lopera.csv',
    'data/test_metadata_lopera.csv'
]

# --- Configuration for the ORIGINAL 'WeightedEmbeddingMLP' experiment ---
CONFIG_WEIGHTED_AVG = {
    "experiment_name": "weighted_avg_benchmark",
    "command_args": {
        "mode": "embedding",
        "embedding_path": "/home/aleph/redlat/walm-time-pooled",
        "lr_weights": "0.001",
        "hidden_size": "1024", # Your chosen best params
        "dropout": "0.4",
        "embedding_layers": [str(i) for i in range(13)], # This needs to be a list of strings
        "parse_weights": True # Tell the script to look for weight analysis
    }
}

# --- Configuration for the NEW 'StatPoolingMLP' experiment ---
CONFIG_STAT_POOLING = {
    "experiment_name": "stat_pooling_benchmark",
    "command_args": {
        "mode": "stat_pooling",
        "word_embedding_pt": "/home/aleph/redlat/lopera_processed_word_embeddings.pt", # !!! UPDATE THIS PATH !!!
        "hidden_size": "1024", # Start with a reasonable default
        "dropout": "0.4",
        "parse_weights": False # The StatPoolingMLP doesn't have layer weights to analyze
    }
}

# --- Fixed parameters shared by both experiments ---
FIXED_PARAMS = {
    "pitch_csv": "data/test_metadata_lopera.csv",
    "timing_csv": "data/test_metadata_lopera.csv",
    "imputation_means": "data/test_metadata_lopera.csv",
    "scaling_params": "data/test_metadata_lopera.csv",
    "classes": ["CN", "AD"],
    "epochs": "100",
    "batch_size": "32",
    "lr": "0.0001",
}


# ==============================================================================
# --- 2. HELPER FUNCTIONS (Unchanged) ---
# ==============================================================================
def parse_benchmark_results(log_file_path, parse_weights=False):
    # ... (The parsing logic is mostly the same, but we add a switch)
    results = {"uar": None, "weights": None}
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            match = re.search(r"Test UAR \(Balanced Accuracy\): (\d\.\d+)", content)
            if match: results["uar"] = float(match.group(1))
    except FileNotFoundError:
        print(f"!!! WARNING: Log file not found at {log_file_path}")
        return None

    if parse_weights:
        try:
            with open(log_file_path, 'r') as f:
                content = f.read()
                weights_section = re.search(r"Final learned importance \(softmax weight\) for each embedding layer:(.*)", content, re.DOTALL)
                if weights_section:
                    weights_str = weights_section.group(1)
                    layer_weights = {}
                    matches = re.findall(r"- Layer\s+(\d+)\s*:\s*(\d\.\d+)", weights_str)
                    for layer_num, weight_val in matches:
                        layer_weights[int(layer_num)] = float(weight_val)
                    if layer_weights: results["weights"] = layer_weights
        except FileNotFoundError: pass
        
    return results


# ==============================================================================
# --- 3. MAIN BENCHMARKING SCRIPT ---
# ==============================================================================
def main(config):
    print(f"--- Starting Benchmark for Experiment: {config['experiment_name']} ---")

    # (Data loading and splitting logic is unchanged)
    df_list = [pd.read_csv(p) for p in ALL_META_PATHS]
    master_df = pd.concat(df_list, ignore_index=True)
    num_patients = master_df['record_id'].nunique()
    print(f"Master dataset size: {len(master_df)} total samples from {num_patients} unique patients.")
    X = master_df.index
    y = master_df['clinical_diagnosis'].values
    sss = StratifiedShuffleSplit(n_splits=NUM_SPLITS, test_size=TEST_SIZE, random_state=42)
    all_split_results = []

    for split_num, (train_index, test_index) in enumerate(sss.split(X, y)):
        print(f"\n--- Processing Split {split_num + 1}/{NUM_SPLITS} ---")

        split_train_df = master_df.iloc[train_index]
        split_test_df = master_df.iloc[test_index]
        temp_train_path = 'data/temp_benchmark_train.csv'
        temp_test_path = 'data/temp_benchmark_test.csv'
        split_train_df.to_csv(temp_train_path, index=False)
        split_test_df.to_csv(temp_test_path, index=False)

        # Use the experiment name for unique directories
        results_dir = f"results/{config['experiment_name']}/split_{split_num + 1}"
        plots_dir = f"plots/{config['experiment_name']}/split_{split_num + 1}"
        save_path = os.path.join(results_dir, "best_model.pth")

        # --- Build the command dynamically from the chosen config ---
        command = [
            "python", "src/train.py",
            "--train_meta", temp_train_path,
            "--val_meta", temp_test_path,
            "--test_meta", temp_test_path,
            "--results_dir", results_dir,
            "--plots_dir", plots_dir,
            "--save_path", save_path,
        ]
        
        # Add shared fixed parameters
        for key, val in FIXED_PARAMS.items():
            if isinstance(val, list): command.extend([f"--{key}", *val])
            else: command.extend([f"--{key}", str(val)])

        # Add experiment-specific parameters
        for key, val in config["command_args"].items():
            if key == "parse_weights": continue # This is for the parser, not train.py
            if isinstance(val, list): command.extend([f"--{key}", *val])
            else: command.extend([f"--{key}", str(val)])
        
        subprocess.run(command, check=True)

        # --- Parse results ---
        mode = config["command_args"]["mode"]
        experiment_name_base = f"{FIXED_PARAMS['classes'][0]}_vs_{FIXED_PARAMS['classes'][1]}_{mode}"
        if mode == 'embedding': # Handle different experiment name conventions
            layers_str = '_'.join(config['command_args']['embedding_layers'])
            experiment_name = f"{experiment_name_base}_layers_{layers_str}"
        else:
            experiment_name = experiment_name_base
            
        log_file = os.path.join(results_dir, f"{experiment_name}_results.txt")
        split_result = parse_benchmark_results(log_file, parse_weights=config["command_args"]["parse_weights"])
        if split_result:
            all_split_results.append(split_result)
            print(f"Split {split_num + 1} UAR: {split_result['uar']:.4f}")

    # --- Final Summary Reporting ---
    print(f"\n\n--- Benchmark Analysis Complete for {config['experiment_name']} ---")
    uar_scores = [res['uar'] for res in all_split_results if res and res['uar'] is not None]
    if uar_scores:
        mean_uar, std_uar = np.mean(uar_scores), np.std(uar_scores)
        print(f"UAR Scores across {NUM_SPLITS} splits: {[f'{s:.4f}' for s in uar_scores]}")
        print(f"--> Final Benchmark UAR: {mean_uar:.4f} ± {std_uar:.4f}\n")
    else:
        print("No UAR scores were collected.")

    if config["command_args"]["parse_weights"]:
        print("--- Learned Weight Analysis Across Splits ---")
        layers_to_check = [0, 3, 7, 11]
        for layer_idx in layers_to_check:
            layer_weights = [res['weights'][layer_idx] for res in all_split_results if res and res.get('weights')]
            if layer_weights:
                mean_w, std_w = np.mean(layer_weights), np.std(layer_weights)
                print(f"Layer {layer_idx} Importance: {mean_w:.4f} ± {std_w:.4f}")
                print(f"  (Values: {[f'{w:.4f}' for w in layer_weights]})")

# ==============================================================================
# --- 4. SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark experiments.")
    parser.add_argument(
        '--experiment', 
        type=str, 
        required=True, 
        choices=['weighted_avg', 'stat_pooling'],
        help="Which experiment configuration to run."
    )
    args = parser.parse_args()

    if args.experiment == 'weighted_avg':
        selected_config = CONFIG_WEIGHTED_AVG
    elif args.experiment == 'stat_pooling':
        selected_config = CONFIG_STAT_POOLING
    else:
        # This case is handled by 'choices' but is good practice
        raise ValueError("Invalid experiment name.")

    main(selected_config)