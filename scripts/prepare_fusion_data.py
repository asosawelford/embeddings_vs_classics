import pandas as pd
import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ================= CONFIGURATION =================
# These should match the features used in your best linguistic model
LINGUISTIC_TASKS = ['CraftIm', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
LINGUISTIC_FEATURES = ["concreteness", "granularity", "mean_num_syll", "mean_phon_neigh", "mean_log_frq"]
# =================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare ALIGNED acoustic and linguistic data for fusion.")
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--features_csv', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='fusion_data')
    parser.add_argument('--classes', nargs='+', default=['AD', 'CN'])
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Prepare Linguistic DataFrame ---
    print("Step 1: Loading and preparing linguistic data...")
    meta_df = pd.read_csv(args.metadata, low_memory=False)
    features_df = pd.read_csv(args.features_csv, low_memory=False)
    df_ling = pd.merge(meta_df[['record_id', 'clinical_diagnosis']], features_df, on='record_id', how='inner')
    
    ling_feature_cols = [col for col in df_ling.columns if any(t in col for t in LINGUISTIC_TASKS) and any(f in col for f in LINGUISTIC_FEATURES)]
    
    # Keep only essential columns
    df_ling_final = df_ling[['record_id', 'clinical_diagnosis'] + ling_feature_cols].copy()
    print(f"  -> Found {len(ling_feature_cols)} linguistic features.")

    # --- Step 2: Prepare Acoustic DataFrame ---
    print("\nStep 2: Processing audio embeddings...")
    acoustic_data = []
    
    # Iterate through all .pt files and extract acoustic features
    for entry in tqdm(os.scandir(args.embeddings_dir), desc="Scanning audio files"):
        if not entry.name.endswith('.pt'):
            continue
            
        record_id = entry.name.split('_')[0]
        
        try:
            data = torch.load(entry.path, map_location='cpu', weights_only=True)
            tensor = data['embeddings']
            if tensor.shape[0] == 0:
                continue
            
            # Simple mean across layers. You can replace this with your WeightedAverage if you have the weights.
            layer_mean = torch.mean(tensor, dim=1) # -> [Words, 768]
            
            mean_vec = torch.mean(layer_mean, dim=0)
            std_vec = torch.std(layer_mean, dim=0) if layer_mean.shape[0] > 1 else torch.zeros_like(mean_vec)
            
            task_vector = torch.cat([mean_vec, std_vec]).numpy()
            
            acoustic_data.append({'record_id': record_id, 'acoustic_vector': task_vector})
        except Exception as e:
            print(f"Warning: Could not process {entry.name}. Error: {e}")
            
    df_acou_raw = pd.DataFrame(acoustic_data)
    
    # Aggregate by patient: if a patient has multiple tasks, average their acoustic vectors
    # This creates one definitive acoustic vector per patient.
    df_acou_final = df_acou_raw.groupby('record_id')['acoustic_vector'].apply(lambda x: np.mean(np.stack(x), axis=0)).reset_index()
    print(f"  -> Processed and aggregated features for {len(df_acou_final)} unique patients.")

    # --- Step 3: THE GRAND MERGE (Guarantees Alignment) ---
    print("\nStep 3: Merging linguistic and acoustic data...")
    # `how='inner'` ensures we only keep patients that exist in BOTH datasets.
    df_fusion = pd.merge(df_ling_final, df_acou_final, on='record_id', how='inner')
    
    df_fusion = df_fusion[df_fusion['clinical_diagnosis'].isin(args.classes)].dropna(subset=['clinical_diagnosis'])
    print(f"  -> Final aligned dataset has {len(df_fusion)} patients.")

    # --- Step 4: Final Split and Save ---
    print("\nStep 4: Performing stratified split and saving data...")
    le = LabelEncoder().fit(df_fusion['clinical_diagnosis'])
    df_fusion['label'] = le.transform(df_fusion['clinical_diagnosis'])

    # Stratified split on the final, merged dataframe
    train_val_df, test_df = train_test_split(df_fusion, test_size=0.15, stratify=df_fusion['label'], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, stratify=train_val_df['label'], random_state=42)
    
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    # Fit imputer and scaler ON TRAINING DATA ONLY
    X_train_ling = train_df[ling_feature_cols].values
    imputer = SimpleImputer(strategy='mean').fit(X_train_ling)
    scaler = StandardScaler().fit(imputer.transform(X_train_ling))

    for split_name, df_split in splits.items():
        print(f"  - Processing and saving '{split_name}' split...")
        
        # Linguistic
        X_ling = df_split[ling_feature_cols].values
        X_ling_processed = scaler.transform(imputer.transform(X_ling))
        
        # Acoustic (already a single vector per patient)
        X_acou = np.stack(df_split['acoustic_vector'].values)
        
        # Labels
        y = df_split['label'].values
        
        # Save
        np.save(os.path.join(args.output_dir, f"{split_name}_linguistic.npy"), X_ling_processed)
        np.save(os.path.join(args.output_dir, f"{split_name}_acoustic.npy"), X_acou)
        np.save(os.path.join(args.output_dir, f"{split_name}_labels.npy"), y)
        
    print("\nFusion data preparation complete!")

if __name__ == "__main__":
    main()



"""
python scripts/prepare_fusion_data.py \
    --metadata "/home/aleph/embeddings_vs_classics/data/metadata_fugu_full_buffer0_emparejado_GLOBAL_additive.csv" \
    --features_csv "/home/aleph/embeddings_vs_classics/data/REDLAT_features.csv" \
    --embeddings_dir "/home/aleph/redlat/embeddings/processed_word_embeddings.pt"
"""