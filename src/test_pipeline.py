import numpy as np
import torch
from pathlib import Path
from data_manager import DataManager, AlzheimerDataset

# --- CONFIGURATION (EDIT THIS) ---
METADATA_PATH = "/home/aleph/embeddings_vs_classics/data/metadata.csv"        # Your metadata.csv
WAVLM_DIR = "/home/aleph/redlat/REDLAT_24-09-25_masked_prepro_wavlm"        # Folder with .npz files
ROBERTA_DIR = "/home/aleph/redlat/REDLAT_24-09-25_transcriptions_gemini_roberta"    # Folder with .npz files
CLASSIC_CSV = "/home/aleph/embeddings_vs_classics/data/REDLAT_features.csv"  # Your classic features csv

# Tasks you want to test
TASKS = ['CraftIm', 'Fugu'] # picking just 2 for testing

# Groups
GROUPS = ['CN', 'AD']
# ---------------------------------

def test_pipeline():
    print("========================================")
    print("      STARTING PIPELINE SANITY CHECK    ")
    print("========================================")

    # 1. Initialize Manager
    print("\n[Step 1] Initializing DataManager...")
    try:
        manager = DataManager(METADATA_PATH, target_groups=GROUPS)
    except Exception as e:
        print(f"❌ Failed to init DataManager: {e}")
        return

    # 2. Simulate a Train/Test Split
    print("\n[Step 2] Simulating a Split (80/20)...")
    all_patients = manager.unique_patients
    n_train = int(len(all_patients) * 0.8)
    
    train_pats = all_patients[:n_train]
    test_pats = all_patients[n_train:]
    
    print(f"   Total Patients: {len(all_patients)}")
    print(f"   Train Patients: {len(train_pats)}")
    print(f"   Test Patients:  {len(test_pats)}")
    
    # Check for leakage
    overlap = set(train_pats).intersection(set(test_pats))
    if overlap:
        print(f"❌ DATA LEAKAGE DETECTED! Overlap: {overlap}")
        return
    else:
        print("✅ No patient overlap between splits.")

    train_df, test_df = manager.split_patients(train_pats, test_pats)

    # ---------------------------------------------------------
    # TEST 3: WavLM Loading (13 layers)
    # ---------------------------------------------------------
    print("\n[Step 3] Testing WavLM Loading...")
    try:
        X_train, y_train, ids, tasks = manager.load_embeddings(
            train_df, WAVLM_DIR, TASKS, model_type='wavlm'
        )
        
        if len(X_train) == 0:
            print("⚠️  No WavLM files found. Check paths/filenames.")
        else:
            print(f"   Loaded {len(X_train)} samples.")
            print(f"   Shape: {X_train.shape}")
            
            # Check Shape
            if X_train.ndim == 3 and X_train.shape[1] == 13 and X_train.shape[2] == 768:
                print("✅ WavLM Shape is correct (Batch, 13, 768).")
            else:
                print(f"❌ WavLM Shape Incorrect! Expected (N, 13, 768), got {X_train.shape}")
                
            # Check Pytorch Dataset Wrapper
            ds = AlzheimerDataset(X_train, y_train, ids, tasks)
            sample = ds[0]
            print(f"   Dataset item keys: {sample.keys()}")
            print(f"   Task in sample: {sample['task']}")

    except Exception as e:
        print(f"❌ WavLM Test Failed: {e}")

    # ---------------------------------------------------------
    # TEST 4: RoBERTa Loading (1 layer)
    # ---------------------------------------------------------
    print("\n[Step 4] Testing RoBERTa Loading...")
    try:
        X_train, y_train, ids, tasks = manager.load_embeddings(
            train_df, ROBERTA_DIR, TASKS, model_type='roberta'
        )
        
        if len(X_train) == 0:
            print("⚠️  No RoBERTa files found.")
        else:
            print(f"   Loaded {len(X_train)} samples.")
            print(f"   Shape: {X_train.shape}")
            
            if X_train.ndim == 2 and X_train.shape[1] == 768:
                print("✅ RoBERTa Shape is correct (Batch, 768).")
            else:
                print(f"❌ RoBERTa Shape Incorrect! Expected (N, 768), got {X_train.shape}")

    except Exception as e:
        print(f"❌ RoBERTa Test Failed: {e}")

    # ---------------------------------------------------------
    # TEST 5: Classic Features (Scaling logic)
    # ---------------------------------------------------------
    print("\n[Step 5] Testing Classic Features (Imputation & Scaling)...")
    try:
        # Note: This assumes your classic CSV has a 'task' column or works with the merge logic
        (X_tr, y_tr, _, _), (X_te, y_te, _, _) = manager.load_classic_features(
            train_df, test_df, CLASSIC_CSV, TASKS, k=5
        )
        
        print(f"   Train Shape: {X_tr.shape}")
        print(f"   Test Shape:  {X_te.shape}")
        
        # Check NaNs
        if np.isnan(X_tr).any() or np.isnan(X_te).any():
            print("❌ NaNs detected after Imputation!")
        else:
            print("✅ No NaNs found.")
            
        # Check Scaling (Train should be mean~0 std~1, Test should be close but not exact)
        tr_mean = np.mean(X_tr)
        tr_std = np.std(X_tr)
        print(f"   Train Mean (should be ~0): {tr_mean:.4f}")
        print(f"   Train Std  (should be ~1): {tr_std:.4f}")
        
        if abs(tr_mean) < 0.1 and abs(tr_std - 1.0) < 0.1:
            print("✅ Scaling looks correct.")
        else:
            print("⚠️ Scaling might be off (or data is very sparse).")

    except Exception as e:
        print(f"❌ Classic Features Test Failed: {e}")
        print("   (Hint: Check if your CSV has 'record_id' and 'task' columns)")

    print("\n========================================")
    print("           TEST COMPLETE                ")
    print("========================================")

if __name__ == "__main__":
    test_pipeline()