import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os


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
                 scaling_params_path, classes_to_load, embedding_layers=None): # embedding_layers is no longer used but kept for compatibility
        
        self.mode = mode
        self.embedding_path = embedding_path
        self.classes_to_load = classes_to_load

        # 1. Load metadata and filter for the binary task
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df[metadata_df['clinical_diagnosis'].isin(self.classes_to_load)]

        # 2. Create labels map
        self.labels_map = {self.classes_to_load[0]: 0, self.classes_to_load[1]: 1}
        
        # (Clinical feature loading commented out as in your example)
        # if self.mode in ['clinical', 'fusion']:
        #     ...
        
        # 3. Create the master list of samples
        self.samples = []
        tasks = ['CraftIm', 'CraftDe' 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
        # tasks = ['Fugu']
        for _, row in metadata_df.iterrows():
            record_id = row['record_id']
            diagnosis = row['clinical_diagnosis']
            for task_name in tasks:
                self.samples.append((record_id, task_name, diagnosis))

        # --- RE-IMPLEMENTING PRE-LOADING STRATEGY for .npy files ---
        if self.mode in ['embedding', 'fusion']:
            self.preloaded_embeddings = {}
            print(f"Pre-loading {len(self.samples)} embedding samples from .npy files...")
            num_failed = 0
            
            for record_id, task_name, _ in tqdm(self.samples, desc="Pre-loading embeddings"):
                sample_key = (record_id, task_name)
                
                if sample_key in self.preloaded_embeddings:
                    continue

                # Your file path structure
                npy_path = os.path.join(self.embedding_path, record_id, f"REDLAT_{record_id}_{task_name}.npy")
                
                try:
                    # --- THIS IS THE FIX ---
                    # Load the .npy file directly without a 'with' statement.
                    embedding_array = np.load(npy_path)
                    
                    # Convert to a flattened torch tensor.
                    embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32)
                    self.preloaded_embeddings[sample_key] = embedding_tensor

                except FileNotFoundError:
                    # Mark sample as invalid if file doesn't exist
                    self.preloaded_embeddings[sample_key] = None
                    num_failed += 1 # Increment failure count
                except Exception as e:
                    # Catch any other loading errors
                    print(f"!!! An unexpected error occurred while loading {npy_path}: {e}. Skipping sample.")
                    self.preloaded_embeddings[sample_key] = None
                    num_failed += 1 # Increment failure count

            total_samples = len(self.samples)
            if total_samples > 0 and (num_failed / total_samples) > 0.9:
                raise RuntimeError(
                    f"FATAL: Pre-loading failed for {num_failed}/{total_samples} samples. "
                    "This is likely due to an incorrect embedding path or file naming convention. "
                    "Please check your --embedding_path and dataloader logic."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_id, task_name, diagnosis = self.samples[idx]
        label = torch.tensor(self.labels_map[diagnosis], dtype=torch.long)
        data_dict = {'label': label, 'record_id': record_id}

        # --- Clinical Feature Loading --- (remains unchanged)
        if self.mode in ['clinical', 'fusion']:
            # ... Your clinical loading logic would go here ...
            pass

        # --- HIGH-PERFORMANCE EMBEDDING RETRIEVAL ---
        if self.mode in ['embedding', 'fusion']:
            sample_key = (record_id, task_name)
            # Retrieve the tensor from RAM
            embedding_tensor = self.preloaded_embeddings.get(sample_key)

            if embedding_tensor is None:
                # This sample was invalid (e.g., file not found), so skip it.
                return None
            
            data_dict['embedding_feats'] = embedding_tensor

        return data_dict