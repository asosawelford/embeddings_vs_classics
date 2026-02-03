import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from data_manager import DataManager

def inspect_features(args):
    print(f"🕵️‍♂️ Inspecting Classic Features (Strict Mode)...")
    if args.task:
        print(f"   Target Task: {args.task}")
    
    # 1. Load Data Manager
    manager = DataManager(args.metadata, target_groups=['CN', 'AD'])
    
    # 2. Mimic the "Strict" Load Logic
    print("   Loading CSV...")
    feat_df = pd.read_csv(args.classic_csv, low_memory=False)
    
    # Create "Lean" Metadata
    lean_meta = manager.metadata[['record_id', 'label_encoded']].copy()
    
    # Merge
    print("   Merging Lean Metadata with Features...")
    merged = pd.merge(lean_meta, feat_df, on='record_id', how='inner')
    
    # --- TASK FILTERING (WIDE FORMAT SUPPORT) ---
    if args.task:
        # Check if we are in Long Format (rows) or Wide Format (columns)
        if 'task' in merged.columns:
            # Long Format Logic
            merged = merged[merged['task'] == args.task]
            print(f"   [Long Format] Samples after filtering rows for '{args.task}': {len(merged)}")
        else:
            # Wide Format Logic (Filter Columns)
            print(f"   [Wide Format] Filtering columns starting with '{args.task}'...")
            
            # Keep Metadata columns
            keep_cols = ['record_id', 'label_encoded', 'site', 'criteria_category', 'clinical_diagnosis']
            # Add Feature columns that match the task
            task_cols = [c for c in merged.columns if c.startswith(args.task)]
            
            if not task_cols:
                print(f"❌ Error: No columns found starting with '{args.task}'. Check capitalization.")
                # Print first 5 cols to help debug
                print(f"   First 5 columns in CSV: {list(merged.columns[2:7])}")
                return
            
            # Select only relevant columns
            cols_to_select = [c for c in merged.columns if c in keep_cols or c in task_cols]
            merged = merged[cols_to_select]
            print(f"   Features kept: {len(task_cols)}")
    # --------------------------------------------

    # 3. Drop known non-feature columns
    cols_to_drop = ['record_id', 'label_encoded', 'task', 'site', 'criteria_category', 'clinical_diagnosis']
    cols_to_drop += ['participant_id', 'subject', 'ID', 'Unnamed: 0']
    
    existing_drop = [c for c in cols_to_drop if c in merged.columns]
    
    # Create X and y
    X_df = merged.drop(columns=existing_drop).select_dtypes(include=[np.number])
    y = merged['label_encoded'].values
    
    feature_names = X_df.columns.tolist()
    X = X_df.values
    
    # Handle NaNs
    X = np.nan_to_num(X)

    if X.shape[1] == 0:
        print("❌ Error: No numeric features found after filtering.")
        return

    print(f"   Analyzing {X.shape[1]} features on {X.shape[0]} samples...")

    # 4. Train Random Forest
    print("\n   Training Random Forest to find leakers...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    # 5. Report
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n🚨 TOP 20 PREDICTIVE FEATURES ({args.task if args.task else 'ALL'}) 🚨")
    print("-" * 60)
    
    for i in range(20):
        if i >= len(indices): break
        idx = indices[i]
        score = importances[idx]
        name = feature_names[idx]
        print(f"{i+1:02d}. {name:<45} | Importance: {score:.4f}")
    
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--classic_csv", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, help="Name of the task to filter (e.g. Fugu)")
    
    args = parser.parse_args()
    inspect_features(args)