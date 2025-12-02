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
    # --- ADD 'tasks_to_load' to the __init__ signature ---
    def __init__(self, metadata_path, clinical_feature_paths, embedding_path, mode, imputation_means_path, 
                 scaling_params_path, classes_to_load, embedding_layers=None, tasks_to_load=None):
        
        self.mode = mode
        self.embedding_path = embedding_path
        self.classes_to_load = classes_to_load

        # 1. Load metadata and filter
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df[metadata_df['clinical_diagnosis'].isin(self.classes_to_load)]

        # 2. Create labels map
        self.labels_map = {self.classes_to_load[0]: 0, self.classes_to_load[1]: 1}
        
        # 3. Create the master list of samples
        self.samples = []
        
        # --- THIS IS THE KEY CHANGE ---
        if tasks_to_load is None:
            # If no specific tasks are provided, default to all of them
            tasks = ['CraftIm', 'CraftDe','Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
            tasks = ['Fugu']
        else:
            # Otherwise, use the list that was passed in
            tasks = tasks_to_load
        
        print(f"Dataloader will load the following tasks: {tasks}")
        # --- END CHANGE ---

        for _, row in metadata_df.iterrows():
            record_id = row['record_id']
            diagnosis = row['clinical_diagnosis']
            for task_name in tasks: # This now uses the dynamic list
                self.samples.append((record_id, task_name, diagnosis))

        # --- PRE-LOADING (Your existing code is correct) ---
        if self.mode in ['embedding', 'fusion']:
            self.preloaded_embeddings = {}
            print(f"Pre-loading {len(self.samples)} embedding samples from .npy files...")
            num_failed = 0
            
            # --- ADD A FLAG TO PRINT THE DEBUG MESSAGE ONLY ONCE ---
            first_fail_reported = False

            for record_id, task_name, _ in tqdm(self.samples, desc="Pre-loading embeddings"):
                sample_key = (record_id, task_name)
                
                if sample_key in self.preloaded_embeddings:
                    continue

                npy_path = os.path.join(self.embedding_path, record_id, f"REDLAT_{record_id}_{task_name}.npy")
                
                try:
                    embedding_array = np.load(npy_path)
                    embedding_tensor = torch.tensor(embedding_array, dtype=torch.float32)
                    self.preloaded_embeddings[sample_key] = embedding_tensor
                except FileNotFoundError:
                    if not first_fail_reported:
                        print(f"\n\n--- DEBUG: File not found at this path ---\n{npy_path}\n--- Please verify path and naming. Further errors will be silent. ---\n")
                        first_fail_reported = True
                    self.preloaded_embeddings[sample_key] = None
                    num_failed += 1
                except Exception as e:
                    self.preloaded_embeddings[sample_key] = None
                    num_failed += 1
            
            # --- Your safety check (this is good to keep) ---
            total_samples = len(self.samples)
            if total_samples > 0 and (num_failed / total_samples) > 0.9:
                raise RuntimeError(
                    f"FATAL: Pre-loading failed for {num_failed}/{total_samples} samples."
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