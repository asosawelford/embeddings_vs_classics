import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import subprocess
import os
import itertools

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Data Splitting Settings ---
VAL_SIZE = 0.10  # 15% of the train+val pool will be for validation
TEST_SIZE = 0.10 # 20% of the total data will be the final hold-out test set

# --- Data Paths ---
ALL_META_PATHS = [
    'data/train_metadata_lopera.csv',
    'data/validation_metadata_lopera.csv',
    'data/test_metadata_lopera.csv'
]
FINAL_DATA_DIR = 'data/final_split' # Directory to save the new splits

# --- Specialist Task Definitions ---
TASK_GROUPS = {
    "video_retelling": ['Fugu'],
    "fluency": ['Phonological', 'Phonological2', 'Semantic', 'Semantic2'],
    "short_story": ['CraftIm']
}

# --- Final Model Parameters ---
# Use the best hyperparameters you found from your tuning benchmark
FINAL_MODEL_PARAMS = {
    "hidden_size": "1024",
    "dropout": "0.4",
    "patience": "20"
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
# --- 2. MAIN SCRIPT ---
# ==============================================================================

def main():
    # --- PART 1: CREATE FINAL, CLEAN DATA SPLITS ---
    print(f"--- 1. Creating Final Data Splits in '{FINAL_DATA_DIR}/' ---")
    os.makedirs(FINAL_DATA_DIR, exist_ok=True)
    
    df_list = [pd.read_csv(p) for p in ALL_META_PATHS if os.path.exists(p)]
    master_df = pd.concat(df_list, ignore_index=True)

    X = master_df.index
    y = master_df['clinical_diagnosis'].values

    # First, split into a development set (train+val) and a final test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    dev_idx, test_idx = next(sss_test.split(X, y))
    
    dev_df = master_df.iloc[dev_idx]
    test_df = master_df.iloc[test_idx]

    # Now, split the development set into a final train set and a final val set
    X_dev = dev_df.index
    y_dev = dev_df['clinical_diagnosis'].values
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=42)
    train_idx, val_idx = next(sss_val.split(X_dev, y_dev))
    
    train_df = dev_df.iloc[train_idx]
    val_df = dev_df.iloc[val_idx]
    
    # Save the three clean files
    final_train_path = os.path.join(FINAL_DATA_DIR, 'final_train_set.csv')
    final_val_path = os.path.join(FINAL_DATA_DIR, 'final_val_set.csv')
    final_test_path = os.path.join(FINAL_DATA_DIR, 'final_test_set.csv')

    train_df.to_csv(final_train_path, index=False)
    val_df.to_csv(final_val_path, index=False)
    test_df.to_csv(final_test_path, index=False)
    
    print(f"  - Final Train Set: {len(train_df)} samples")
    print(f"  - Final Val Set:   {len(val_df)} samples")
    print(f"  - Final Test Set:  {len(test_df)} samples (HOLD-OUT)")

    # --- PART 2: TRAIN SPECIALIST MODELS ON THE CLEAN SPLITS ---
    print("\n--- 2. Training Specialist Models ---")

    for group_name, tasks_in_group in TASK_GROUPS.items():
        print(f"\n{'='*15} Training Specialist: {group_name.upper()} {'='*15}")
        
        # Define unique paths for this specialist model
        model_dir = "models/specialists"
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f"{group_name}_model.pth")
        
        results_dir = f"results/specialist_training/{group_name}"
        plots_dir = f"plots/specialist_training/{group_name}"

        command = [
            "python", "src/train.py",
            # Use the clean data splits
            "--train_meta", final_train_path,
            "--val_meta", final_val_path,
            "--test_meta", final_val_path, # We evaluate on val set, final test set is untouched
            # Paths and directories
            "--results_dir", results_dir,
            "--plots_dir", plots_dir,
            "--save_path", save_path,
            # The tasks for this specialist
            "--tasks", *tasks_in_group,
        ]
        
        # Add fixed and hyper-parameters
        for key, val in {**FIXED_TRAINING_PARAMS, **FINAL_MODEL_PARAMS}.items():
            if isinstance(val, list):
                command.extend([f"--{key}", *val])
            else:
                command.extend([f"--{key}", str(val)])
        
        command.extend(["--embedding_layers", *[str(l) for l in range(13)]])

        try:
            subprocess.run(command, check=True)
            print(f"✅ Successfully trained and saved model for '{group_name}' at {save_path}")
        except subprocess.CalledProcessError as e:
            print(f"!!! ERROR training specialist '{group_name}' !!!")
            print(e.stderr)
            break # Stop if one model fails to train

    print("\n--- All specialist models trained successfully! ---")

if __name__ == "__main__":
    main()