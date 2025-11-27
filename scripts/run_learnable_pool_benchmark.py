import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
import os
import re

# ==============================================================================
# --- 1. EXPERIMENT CONFIGURATION ---
# ==============================================================================

# --- The script we are going to benchmark ---
SCRIPT_TO_RUN = "src/train_learnable_pool.py"

# --- Main data file and metadata paths ---
# !!! UPDATE THIS PATH to your main processed data file !!!
DATA_PT_PATH = "/home/aleph/redlat/lopera_processed_word_embeddings.pt"

ALL_META_PATHS = [
    'data/train_metadata_lopera.csv',
    'data/validation_metadata_lopera.csv',
    'data/test_metadata_lopera.csv'
]

# --- Benchmark Settings ---
NUM_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Hyperparameters for the LearnableStatPoolingMLP model ---
MODEL_PARAMS = {
    "lr": "0.0001",
    "lr_weights": "0.001",
    "epochs": "100",
    "batch_size": "16",
    "hidden_size": "1024",
    "patience": "20",
    "classes": ["AD", "CN"],
}

# ==============================================================================
# --- 2. HELPER FUNCTION TO PARSE RESULTS ---
# ==============================================================================

def parse_benchmark_results(log_file_path):
    """
    Parses the final results file for the Test UAR and the learned layer weights.
    """
    results = {"uar": None, "weights": None}
    
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

            # Find the UAR score
            uar_match = re.search(r"Test UAR \(Balanced Accuracy\):\s*(\d\.\d+)", content)
            if uar_match:
                results["uar"] = float(uar_match.group(1))

            # Find the layer weights section
            weights_section_match = re.search(r"Learned Layer Importance Analysis.*Final importance.*:", content, re.DOTALL)
            if weights_section_match:
                weights_content = weights_section_match.group(0) # Get the whole section
                layer_weights = {}
                # Find all layer weight lines within that section
                weight_matches = re.findall(r"-\s*Layer\s+(\d+)\s*:\s*(\d\.\d+)", weights_content)
                for layer_num, weight_val in weight_matches:
                    layer_weights[int(layer_num)] = float(weight_val)
                if layer_weights:
                    results["weights"] = layer_weights
    
    except FileNotFoundError:
        print(f"!!! WARNING: Log file not found at {log_file_path}")
        return None
        
    return results

# ==============================================================================
# --- 3. MAIN BENCHMARKING SCRIPT ---
# ==============================================================================
def main():
    EXPERIMENT_BASE_NAME = "LearnablePool_ADvCN_Benchmark"
    print(f"--- Starting Benchmark for Experiment: {EXPERIMENT_BASE_NAME} ---")

    # 1. Load and concatenate all metadata into a master dataframe
    df_list = [pd.read_csv(p) for p in ALL_META_PATHS]
    # We just combine the files first. The next step will handle creating the unique patient list.
    master_df = pd.concat(df_list, ignore_index=True)
    
    # 2. Create a unique list of PATIENTS for stratification
    # This line correctly creates a unique patient list for the splitter.
    patient_df = master_df.drop_duplicates(subset=['record_id']).set_index('record_id')
    
    print(f"Master dataset has {len(master_df)} total samples from {len(patient_df)} unique patients.")
    
    # 2. Setup Stratified Splitting
    X = patient_df.index # Patient IDs
    y = patient_df['clinical_diagnosis'] # Patient labels
    sss = StratifiedShuffleSplit(n_splits=NUM_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    all_split_results = []

    # 3. Loop through each split
    for split_num, (train_patient_idx, test_patient_idx) in enumerate(sss.split(X, y)):
        split_name = f"split_{split_num + 1}"
        print(f"\n--- Processing {split_name}/{NUM_SPLITS} ---")

        # Get the patient IDs for this split's train and test sets
        train_patient_ids = X[train_patient_idx]
        test_patient_ids = X[test_patient_idx]

        # Create the actual train/test dataframes from the master list of samples
        split_train_df = master_df[master_df['record_id'].isin(train_patient_ids)]
        split_test_df = master_df[master_df['record_id'].isin(test_patient_ids)]

        # Save temporary metadata files for this split
        temp_dir = 'data/temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_train_path = os.path.join(temp_dir, 'temp_benchmark_train.csv')
        temp_test_path = os.path.join(temp_dir, 'temp_benchmark_test.csv')
        split_train_df.to_csv(temp_train_path, index=False)
        split_test_df.to_csv(temp_test_path, index=False)
        
        print(f"Split {split_num + 1}: {len(train_patient_ids)} train patients, {len(test_patient_ids)} test patients.")

        # 4. Build and Run the Subprocess Command
        experiment_name_for_split = f"{EXPERIMENT_BASE_NAME}_{split_name}"
        results_dir = f"results/{EXPERIMENT_BASE_NAME}/{split_name}"
        plots_dir = f"plots/{EXPERIMENT_BASE_NAME}/{split_name}"

        command = [
            "python", SCRIPT_TO_RUN,
            "--data_path", DATA_PT_PATH,
            "--train_meta", temp_train_path,
            "--val_meta", temp_test_path,    # Using test split for validation in this fold
            "--test_meta", temp_test_path,
            "--results_dir", results_dir,
            "--plots_dir", plots_dir,
            "--experiment_name", experiment_name_for_split, # Pass the specific name
        ]
        
        # Add all model hyperparameters
        for key, val in MODEL_PARAMS.items():
            if isinstance(val, list):
                command.extend([f"--{key}", *val])
            else:
                command.extend([f"--{key}", str(val)])
        
        print("Running command:", ' '.join(command))
        subprocess.run(command, check=True)

        # 5. Parse results from the output file
        log_file = os.path.join(results_dir, f"{experiment_name_for_split}_results.txt")
        split_result = parse_benchmark_results(log_file)
        
        if split_result and split_result["uar"] is not None:
            all_split_results.append(split_result)
            print(f"✅ Split {split_num + 1} UAR: {split_result['uar']:.4f}")
        else:
            print(f"❌ Could not parse results for split {split_num + 1}.")


    # 6. Final Summary Reporting
    print(f"\n\n--- Benchmark Analysis Complete for {EXPERIMENT_BASE_NAME} ---")
    
    uar_scores = [res['uar'] for res in all_split_results if res and res.get('uar') is not None]
    if uar_scores:
        mean_uar, std_uar = np.mean(uar_scores), np.std(uar_scores)
        print(f"UAR Scores across {len(uar_scores)} splits: {[f'{s:.4f}' for s in uar_scores]}")
        print(f"--> Final Benchmark UAR: {mean_uar:.4f} ± {std_uar:.4f}\n")
    else:
        print("No valid UAR scores were collected across all splits.")

    # Analyze layer weights across all splits
    print("--- Average Learned Weight Analysis Across Splits ---")
    # Collect all weights into a DataFrame for easy aggregation
    all_weights = [res['weights'] for res in all_split_results if res and res.get('weights')]
    if all_weights:
        weights_df = pd.DataFrame(all_weights)
        mean_weights = weights_df.mean().sort_values(ascending=False)
        
        print("Average Layer Importance (Mean ± Std Dev):")
        for layer_idx, mean_w in mean_weights.items():
            std_w = weights_df[layer_idx].std()
            print(f"  - Layer {layer_idx:<2}: {mean_w:.4f} ± {std_w:.4f}")
    else:
        print("No layer weights were collected.")

if __name__ == "__main__":
    main()