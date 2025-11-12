import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings # For silencing the pandas warning


def collate_fn(batch, fixed_clinical_len=None):
    """
    Custom collate function. Now handles padding clinical features to a fixed global length.
    
    Args:
        batch (list): A list of sample dictionaries.
        fixed_clinical_len (int, optional): The fixed length to pad clinical features to.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    record_ids = [item['record_id'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    batched_dict = {'label': labels, 'record_id': record_ids}

    has_clinical = 'clinical_feats' in batch[0]
    has_embedding = 'embedding_feats' in batch[0]

    if has_clinical:
        if fixed_clinical_len is None:
            raise ValueError("fixed_clinical_len must be provided for batches with clinical features.")
            
        # Manually pad to the fixed global length
        padded_clinical = torch.zeros(len(batch), fixed_clinical_len)
        for i, item in enumerate(batch):
            feat_tensor = item['clinical_feats']
            length = feat_tensor.shape[0]
            padded_clinical[i, :length] = feat_tensor
            
        batched_dict['clinical_feats'] = padded_clinical

    if has_embedding:
        embedding_feats = torch.stack([item['embedding_feats'] for item in batch])
        batched_dict['embedding_feats'] = embedding_feats

    return batched_dict


class PatientTaskDataset(Dataset):
    def __init__(self, metadata_path, clinical_feature_paths, embedding_path, mode, imputation_means_path, 
                 scaling_params_path, classes_to_load, embedding_layers=[0, 8, 11]): # <--- NEW PARAMETER
        """
        Args:
            scaling_params_path (str): Path to the JSON file with pre-computed scaling parameters.
        """
        self.mode = mode
        self.embedding_path = embedding_path
        self.embedding_layers = embedding_layers
        self.classes_to_load = classes_to_load

        # 1. Load metadata AND FILTER FOR BINARY TASK
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df[metadata_df['clinical_diagnosis'].isin(self.classes_to_load)]

        # 2. DYNAMICALLY CREATE LABELS_MAP FOR BINARY 0/1
        self.labels_map = {self.classes_to_load[0]: 0, self.classes_to_load[1]: 1}
        
        # 3. Pre-load clinical feature files for fast lookup
        self.clinical_features = {}
        for name, path in clinical_feature_paths.items():
            df = pd.read_csv(path).set_index('id')
            self.clinical_features[name] = df
            
        # 4. Load the pre-computed means for imputation
        with open(imputation_means_path, 'r') as f:
            self.imputation_means = json.load(f)

        # 5. Load the pre-computed scaling parameters <--- NEW
        with open(scaling_params_path, 'r') as f:
            scaling_data = json.load(f)
            self.scaling_means = np.array(scaling_data['mean'], dtype=np.float32)
            self.scaling_stds = np.array(scaling_data['std'], dtype=np.float32)
            self.expected_clinical_dim = scaling_data['feature_dim']
        
        # 6. Create the master list of (patient, task) samples
        self.samples = []
        tasks = ['CraftDe', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
        for _, row in metadata_df.iterrows():
            record_id = row['record_id']
            diagnosis = row['clinical_diagnosis']
            for task_name in tasks:
                self.samples.append((record_id, task_name, diagnosis))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_id, task_name, diagnosis = self.samples[idx]
        label = torch.tensor(self.labels_map[diagnosis], dtype=torch.long)
        data_dict = {'label': label, 'record_id': record_id}

        # --- Clinical Feature Loading, Imputation, and SCALING ---
        if self.mode in ['clinical', 'fusion']:
            all_task_feats = []
            for df_name, df in self.clinical_features.items():
                try:
                    patient_row = df.loc[record_id]
                    task_cols = [col for col in patient_row.index if col.startswith(f'{task_name}__')]
                    
                    feature_series = patient_row[task_cols]
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        imputed_series = feature_series.fillna(self.imputation_means).infer_objects(copy=False)
                    
                    if imputed_series.isnull().any():
                         imputed_series = imputed_series.fillna(0.0) 

                    feature_values = imputed_series.values
                    all_task_feats.append(feature_values)

                except KeyError:
                    # If a record_id/task_name is not found in a specific clinical feature CSV,
                    # this sample cannot be complete.
                    print(f"!!! ERROR: Data missing for {record_id} in {df_name}.csv for task {task_name}. Skipping sample.")
                    return None # Return None if sample is incomplete

            try:
                # Concatenate all features for the current sample
                clinical_tensor_unscaled_raw = np.concatenate(all_task_feats)
                
                # --- NEW: PAD TO EXPECTED DIMENSION HERE ---
                if clinical_tensor_unscaled_raw.shape[0] < self.expected_clinical_dim:
                    clinical_tensor_unscaled = torch.tensor(
                        np.pad(clinical_tensor_unscaled_raw, (0, self.expected_clinical_dim - clinical_tensor_unscaled_raw.shape[0]), 'constant', constant_values=0),
                        dtype=torch.float32
                    )
                elif clinical_tensor_unscaled_raw.shape[0] > self.expected_clinical_dim:
                    # This case means the `compute_scaling_params.py` got a smaller max_length
                    # than what's actually present. It implies an issue in how max_length was determined.
                    print(f"!!! CRITICAL WARNING: Sample {record_id}, task {task_name} has more clinical features ({clinical_tensor_unscaled_raw.shape[0]}) than expected max ({self.expected_clinical_dim}). Truncating to avoid error. Investigate data consistency!")
                    clinical_tensor_unscaled = torch.tensor(clinical_tensor_unscaled_raw[:self.expected_clinical_dim], dtype=torch.float32)
                else:
                    clinical_tensor_unscaled = torch.tensor(clinical_tensor_unscaled_raw, dtype=torch.float32)

                # --- Apply Scaling ---
                # Ensure self.scaling_means and self.scaling_stds are numpy arrays of correct shape
                clinical_tensor = (clinical_tensor_unscaled - torch.tensor(self.scaling_means)) / torch.tensor(self.scaling_stds)
                data_dict['clinical_feats'] = clinical_tensor

            except ValueError as e:
                print(f"!!! FATAL ERROR during clinical feature processing for {record_id}, {task_name}: {e}")
                raise e

        # --- Embedding Feature Loading ---
        if self.mode in ['embedding', 'fusion']:
            npz_path = f"{self.embedding_path}/REDLAT_{record_id}_{task_name}.npz"
            try:
                loaded_layer_arrays = []
                with np.load(npz_path) as data:
                    for layer_idx in self.embedding_layers:
                        layer_key = f'layer_{layer_idx}'
                        if layer_key in data:
                            loaded_layer_arrays.append(data[layer_key])
                        else:
                            print(f"!!! FATAL ERROR: Key '{layer_key}' not found in {npz_path}. Skipping sample.")
                            return None # This sample is invalid, return None

                stacked_layers = np.stack(loaded_layer_arrays, axis=0)
                time_averaged_layers = np.mean(stacked_layers, axis=-1) # time-pooling
                embedding_tensor = torch.tensor(time_averaged_layers.flatten(), dtype=torch.float32)
                data_dict['embedding_feats'] = embedding_tensor

            except FileNotFoundError:
                print(f"!!! ERROR: Embedding file not found at {npz_path}. Skipping sample.")
                return None
            except Exception as e:
                print(f"!!! An unexpected error occurred while loading embeddings for {npz_path}: {e}. Skipping sample.")
                return None

        return data_dict
