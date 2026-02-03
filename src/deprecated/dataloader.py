import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
from pathlib import Path


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

# This is a new collate function specifically for sequences of variable length.
def pad_collate_fn(batch):
    """
    Pads sequences to the length of the longest sequence in a batch.
    
    Args:
        batch (list of dicts): A list where each dict is an output from WordSequenceDataset.
                               Each dict must contain 'embedding_feats'.
    
    Returns:
        A dictionary with padded embeddings, original lengths, labels, and record_ids.
    """
    # Filter out None items, if any
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # --- 1. Get original lengths of each sequence ---
    # This is crucial for the model to ignore padding during calculations.
    lengths = torch.tensor([item['embedding_feats'].shape[0] for item in batch])

    # --- 2. Pad the embedding sequences ---
    # pad_sequence expects a list of tensors
    sequences = [item['embedding_feats'] for item in batch]
    # batch_first=True makes the output shape (batch_size, seq_len, num_layers, features)
    padded_embeddings = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # --- 3. Stack labels and gather record_ids ---
    labels = torch.stack([item['label'] for item in batch])
    record_ids = [item['record_id'] for item in batch]
    
    return {
        'embedding_feats': padded_embeddings,
        'lengths': lengths,
        'label': labels,
        'record_id': record_ids
    }


class WordSequenceDataset(Dataset):
    """
    Dataset to load the pre-processed word sequence tensors from a single .pt file.
    It matches embeddings with labels from a metadata CSV.
    """
    def __init__(self, metadata_path, word_embedding_data, classes_to_load):
        
        self.classes_to_load = classes_to_load

        # 1. Use the pre-loaded data directly
        # The data is expected to be a dictionary: (record_id, task) -> tensor
        self.embedding_map = word_embedding_data

        # 2. Load metadata to get labels and create samples
        metadata_df = pd.read_csv(metadata_path)
        metadata_df = metadata_df[metadata_df['clinical_diagnosis'].isin(self.classes_to_load)]
        self.labels_map = {self.classes_to_load[0]: 0, self.classes_to_load[1]: 1}

        self.samples = []
        print("Matching metadata with loaded embeddings...")
        for _, row in metadata_df.iterrows():
            record_id = row['record_id']
            # We assume 'task' column exists in metadata, or we can iterate through all possible tasks
            # For now, let's assume a 'task' column exists. If not, this needs adjustment.
            
            # --- This part is flexible depending on your metadata ---
            # Simplified: Assuming you have a 'task' column in your metadata.csv
            if 'task' in row:
                tasks = [row['task']]
            else: # If no task column, check against all tasks for that ID
                tasks = ['CraftIm', 'Semantic', 'Fugu'] # Or any other relevant tasks

            for task in tasks:
                if (record_id, task) in self.embedding_map:
                    self.samples.append({
                        'record_id': record_id,
                        'task': task,
                        'label': self.labels_map[row['clinical_diagnosis']]
                    })

        print(f"Successfully created {len(self.samples)} samples with corresponding labels.")
        if len(self.samples) == 0:
            raise ValueError("No matching samples found between metadata and embedding file. Check record_id/task names.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        record_id = sample_info['record_id']
        task = sample_info['task']
        
        # Retrieve the embedding tensor
        # Shape: [Num_Words, 13, 768]
        embedding_tensor = self.embedding_map.get((record_id, task))
        
        if embedding_tensor is None or embedding_tensor.shape[0] == 0:
            return None

        return {
            'embedding_feats': embedding_tensor,
            'label': torch.tensor(sample_info['label'], dtype=torch.long),
            'record_id': record_id,
        }

class DiskEmbeddingsDataset(Dataset):
    def __init__(self, metadata_df, embeddings_dir, classes, pos_filter=None):
        """
        Args:
            metadata_df (pd.DataFrame): Must contain 'record_id' and 'clinical_diagnosis'.
            embeddings_dir (str): Folder containing the .pt files.
            classes (list): e.g., ['AD', 'CN'].
            pos_filter (list, optional): e.g., ['NOUN', 'VERB']. If None, uses all words.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.pos_filter = pos_filter
        
        # Filter metadata to only keep relevant classes
        self.meta = metadata_df[metadata_df['clinical_diagnosis'].isin(classes)].copy()
        self.label_map = {classes[0]: 0, classes[1]: 1} # Binary
        
        # --- INDEXING ---
        # We need to map record_id -> list of file paths
        # Filename format expected: "{record_id}_{task}_embeddings.pt"
        self.file_map = {}
        
        # Scan directory once
        print(f"Indexing .pt files in {embeddings_dir}...")
        all_files = list(self.embeddings_dir.glob("*.pt"))
        
        for fpath in all_files:
            # Parse ID from filename (assumes id is the first part before the first underscore)
            # Adjust logic if your IDs contain underscores!
            # Current assumption: "001_task_embeddings.pt" -> id "001"
            fname = fpath.name
            possible_id = fname.split('_')[0] 
            
            if possible_id not in self.file_map:
                self.file_map[possible_id] = []
            self.file_map[possible_id].append(fpath)
            
        # Filter samples: Keep only those that have at least one audio file
        self.samples = []
        missing = 0
        for _, row in self.meta.iterrows():
            rid = str(row['record_id'])
            lbl = self.label_map[row['clinical_diagnosis']]
            
            if rid in self.file_map:
                # A patient might have multiple tasks (files). 
                # We treat each FILE as a training sample.
                for fpath in self.file_map[rid]:
                    self.samples.append({
                        'record_id': rid,
                        'label': lbl,
                        'path': fpath
                    })
            else:
                missing += 1
                
        print(f"Dataset ready. {len(self.samples)} audio samples found. ({missing} patients missing files)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        path = item['path']
        label = item['label']
        
        try:
            # LOAD FROM DISK
            data = torch.load(path, weights_only=True)
            tensor = data['embeddings'] # [Words, 13, 768]
            metadata = data['metadata']
            
            # OPTIONAL: POS Filtering
            if self.pos_filter:
                indices = [i for i, m in enumerate(metadata) if m['pos'] in self.pos_filter]
                if not indices:
                    # No matching words found (e.g., no Verbs in this file)
                    # Return a dummy zero tensor to avoid crashing
                    return None 
                tensor = tensor[indices]

            return {
                'embedding_feats': tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'record_id': item['record_id']
            }
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
