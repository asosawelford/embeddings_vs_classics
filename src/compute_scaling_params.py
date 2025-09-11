import pandas as pd
import numpy as np
import json
import argparse
import warnings # For silencing pandas warnings

def compute_and_save_scaling_params(train_metadata_path, clinical_feature_paths, imputation_means_path, output_path):
    """
    Calculates the mean and standard deviation for each clinical feature column
    using only the training data AFTER imputation, and saves the results to a JSON file.

    Args:
        train_metadata_path (str): Path to the training metadata CSV.
        clinical_feature_paths (dict): Dictionary mapping feature names to their CSV paths.
        imputation_means_path (str): Path to the JSON file with pre-computed imputation means.
        output_path (str): Path to save the output JSON file.
    """
    print(f"Loading training metadata from: {train_metadata_path}")
    train_df_meta = pd.read_csv(train_metadata_path)
    train_patient_ids = train_df_meta['record_id'].tolist()
    
    print(f"Loading imputation means from: {imputation_means_path}")
    with open(imputation_means_path, 'r') as f:
        imputation_means = json.load(f)

    print("Collecting all clinical features and determining max length...")
    
    tasks = ['CraftDe', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
    
    # --- Step 1: Determine the maximum feature length ---
    # This loop will iterate once just to find the max length
    max_clinical_feature_length = 0
    temp_clinical_features_list = [] # Temporary list to store for the second pass
    
    for _, row in train_df_meta.iterrows():
        record_id = row['record_id']
        for task_name in tasks:
            current_sample_feats = []
            for df_name, df_path in clinical_feature_paths.items():
                df = pd.read_csv(df_path).set_index('id')
                try:
                    patient_row = df.loc[record_id]
                    task_cols = [col for col in patient_row.index if col.startswith(f'{task_name}__')]
                    feature_series = patient_row[task_cols]
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        imputed_series = feature_series.fillna(imputation_means).infer_objects(copy=False)
                    
                    if imputed_series.isnull().any():
                         imputed_series = imputed_series.fillna(0.0)
                    
                    current_sample_feats.append(imputed_series.values)
                except KeyError:
                    # If a patient-task combo is incomplete, we cannot use it for feature collection
                    # This implies you have some missing data points in your clinical feature CSVs
                    # for certain patient-task combinations. This needs to be consistent.
                    # Or, some patients simply don't have data for some tasks.
                    print(f"Skipping incomplete clinical feature set for {record_id} task {task_name} from {df_name}. Not all features found.")
                    current_sample_feats = [] # Clear if incomplete
                    break # Stop processing for this patient-task combo
            
            if current_sample_feats: # Only if all clinical feature types were found for this sample
                concatenated_features = np.concatenate(current_sample_feats)
                temp_clinical_features_list.append(concatenated_features)
                if concatenated_features.shape[0] > max_clinical_feature_length:
                    max_clinical_feature_length = concatenated_features.shape[0]

    if not temp_clinical_features_list:
        raise ValueError("No complete clinical feature samples found for scaling calculation. Check data consistency.")

    print(f"Determined maximum clinical feature length: {max_clinical_feature_length}")

    # --- Step 2: Pad all feature vectors to the maximum length and stack ---
    full_train_clinical_features_padded = []
    for feature_vector in temp_clinical_features_list:
        if feature_vector.shape[0] < max_clinical_feature_length:
            # Pad with zeros to the max length
            padded_vector = np.pad(feature_vector, (0, max_clinical_feature_length - feature_vector.shape[0]), 'constant', constant_values=0)
            full_train_clinical_features_padded.append(padded_vector)
        else:
            full_train_clinical_features_padded.append(feature_vector)
            
    # Now all arrays in the list should have the same shape
    all_features_np = np.stack(full_train_clinical_features_padded)
    
    # Calculate mean and std for each feature column
    scaling_means = np.mean(all_features_np, axis=0)
    scaling_stds = np.std(all_features_np, axis=0)

    scaling_stds[scaling_stds == 0] = 1.0 # Avoid division by zero

    scaling_params = {
        'mean': scaling_means.tolist(),
        'std': scaling_stds.tolist(),
        'feature_dim': all_features_np.shape[1] # This will now be max_clinical_feature_length
    }

    print(f"Calculated scaling parameters for {scaling_params['feature_dim']} features.")
    
    with open(output_path, 'w') as f:
        json.dump(scaling_params, f, indent=4)
        
    print(f"✅ Successfully saved imputation means to: {output_path}") # This print was for means, should be scaling params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute feature scaling parameters (mean/std).")
    parser.add_argument('--train_meta', type=str, required=True, help='Path to train_metadata.csv')
    parser.add_argument('--pitch_csv', type=str, required=True, help='Path to pitch_features.csv')
    parser.add_argument('--timing_csv', type=str, required=True, help='Path to timing_features.csv')
    parser.add_argument('--imputation_means', type=str, required=True, help='Path to imputation_means.json')
    parser.add_argument('--output_json', type=str, default='scaling_params.json', help='Path for the output JSON file.')
    
    args = parser.parse_args()

    clinical_paths = {
        'pitch': args.pitch_csv,
        'timing': args.timing_csv
    }

    compute_and_save_scaling_params(
        train_metadata_path=args.train_meta,
        clinical_feature_paths=clinical_paths,
        imputation_means_path=args.imputation_means,
        output_path=args.output_json
    )
