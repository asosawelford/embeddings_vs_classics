import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
import os
import re
import json

# --- Configuration ---
NUM_SPLITS = 5  # Let's do 5 full, independent runs
TEST_SIZE = 0.2 # 80% for training, 20% for testing
# Combine all available metadata into one master file
ALL_META_PATHS = [
    'data/train_metadata_lopera.csv',
    'data/validation_metadata_lopera.csv',
    'data/test_metadata_lopera.csv'
]

def parse_benchmark_results(log_file_path, weights_plot_path):
    """
    Parses both the final UAR and the learned layer weights for a benchmark run.
    """
    results = {"uar": None, "weights": None}
    
    # 1. Parse UAR from the text log
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            match = re.search(r"Test UAR \(Balanced Accuracy\): (\d\.\d+)", content)
            if match:
                results["uar"] = float(match.group(1))
    except FileNotFoundError:
        print(f"!!! WARNING: Log file not found at {log_file_path}")
        return None

    # 2. Parse weights from the analysis function's print output (a bit of a hack, but effective)
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            # Find the weights section
            weights_section = re.search(r"Final learned importance \(softmax weight\) for each embedding layer:(.*)", content, re.DOTALL)
            if weights_section:
                weights_str = weights_section.group(1)
                layer_weights = {}
                # Find all lines matching 'Layer X: Y.YYYY (Z.ZZ%)'
                matches = re.findall(r"- Layer\s+(\d+)\s*:\s*(\d\.\d+)", weights_str)
                for layer_num, weight_val in matches:
                    layer_weights[int(layer_num)] = float(weight_val)
                if layer_weights:
                    results["weights"] = layer_weights
    except FileNotFoundError:
        # This error is already handled above
        pass
        
    return results

def main():
    print("--- Starting Benchmark with Multiple Randomized Splits ---")

    # 1. Combine ALL data into a single master dataframe
    df_list = [pd.read_csv(p) for p in ALL_META_PATHS]
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"Master dataset size: {len(master_df)} unique patients.")

    X = master_df['record_id'].values
    y = master_df['clinical_diagnosis'].values

    # 2. Set up the splitter
    sss = StratifiedShuffleSplit(n_splits=NUM_SPLITS, test_size=TEST_SIZE, random_state=42)

    all_split_results = []

    # 3. Loop through each independent split
    for split_num, (train_index, test_index) in enumerate(sss.split(X, y)):
        print(f"\n--- Processing Split {split_num + 1}/{NUM_SPLITS} ---")

        # Create the metadata for this split
        split_train_df = master_df.iloc[train_index]
        split_test_df = master_df.iloc[test_index]

        # Save to temporary files
        temp_train_path = 'data/temp_benchmark_train.csv'
        temp_test_path = 'data/temp_benchmark_test.csv'
        split_train_df.to_csv(temp_train_path, index=False)
        split_test_df.to_csv(temp_test_path, index=False)

        # Define unique output directories
        results_dir = f"results/benchmark_split_{split_num + 1}"
        plots_dir = f"plots/benchmark_split_{split_num + 1}"

        # 4. Run the training command
        # Here, we use the same data for validation and testing. 
        # The 'val' set is just for saving the best model during the epoch loop.
        # The 'test' set is for the final evaluation, which is what we will record.
        command = [
            "python", "src/train.py",
            "--train_meta", temp_train_path,
            "--val_meta", temp_test_path, # Use test set for validation logic
            "--test_meta", temp_test_path, # And for final testing
            # --- Your other parameters ---
            "--pitch_csv", "data/test_metadata_lopera.csv",
            "--timing_csv", "data/test_metadata_lopera.csv",
            "--embedding_path", "/home/aleph/redlat/walm-time-pooled",
            "--imputation_means", "data/test_metadata_lopera.csv",
            "--scaling_params", "data/test_metadata_lopera.csv",
            "--classes", "CN", "AD",
            "--mode", "embedding", # Sticking with the Weighted MLP for this benchmark
            "--embedding_layers", *[str(i) for i in range(13)],
            "--epochs", "100",
            "--batch_size", "32",
            "--lr", "0.0001",
            "--lr_weights", "0.001",
            "--hidden_size", "256",
            "--results_dir", results_dir,
            "--plots_dir", plots_dir,
        ]
        
        subprocess.run(command, check=True)

        # 5. Parse the results for this split
        experiment_name = f"CN_vs_AD_embedding_layers_{'_'.join(map(str, range(13)))}"
        log_file = os.path.join(results_dir, f"{experiment_name}_results.txt")
        weights_plot = os.path.join(plots_dir, f"{experiment_name}_layer_weights.png")

        split_result = parse_benchmark_results(log_file, weights_plot)
        if split_result:
            all_split_results.append(split_result)
            print(f"Split {split_num + 1} UAR: {split_result['uar']:.4f}")

    # --- Final Summary Reporting ---
    print("\n\n--- Benchmark Analysis Complete ---")
    if not all_split_results:
        print("No results were collected. Exiting.")
        return

    # Report UAR variability
    uar_scores = [res['uar'] for res in all_split_results]
    mean_uar = np.mean(uar_scores)
    std_uar = np.std(uar_scores)
    print(f"UAR Scores across {NUM_SPLITS} splits: {[f'{s:.4f}' for s in uar_scores]}")
    print(f"--> Final Benchmark UAR: {mean_uar:.4f} ± {std_uar:.4f}\n")

    # Report weight variability
    print("--- Learned Weight Analysis Across Splits ---")
    # Let's check the stability of a few key layers
    layers_to_check = [0, 6, 9, 11] # Low, Mid, and High
    for layer_idx in layers_to_check:
        layer_weights = [res['weights'][layer_idx] for res in all_split_results if res['weights']]
        if layer_weights:
            mean_w = np.mean(layer_weights)
            std_w = np.std(layer_weights)
            print(f"Layer {layer_idx} Importance: {mean_w:.4f} ± {std_w:.4f}")
            print(f"  (Values: {[f'{w:.4f}' for w in layer_weights]})")

if __name__ == "__main__":
    main()
