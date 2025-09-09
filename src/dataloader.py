# In src/dataloader.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PatientTaskDataset(Dataset):
    def __init__(self, metadata_path, clinical_feature_paths, embedding_path, mode, embedding_layers=[0, 8, 11]):
        """
        Args:
            metadata_path (str): Path to the train, validation, or test metadata CSV.
            clinical_feature_paths (dict): Dict like {'pitch': 'path/to/pitch.csv', 'timing': 'path/to/timing.csv'}.
            embedding_path (str): Path to the folder containing WavLM .npz files.
            mode (str): One of 'clinical', 'embedding', or 'fusion'.
            embedding_layers (list): List of WavLM layer indices to use (0-12).
        """
        self.mode = mode
        self.embedding_path = embedding_path
        self.embedding_layers = embedding_layers

        # 1. Load metadata
        metadata_df = pd.read_csv(metadata_path)
        self.labels_map = {'CN': 0, 'FTD': 1, 'AD': 2}
        
        # 2. Pre-load clinical feature files for fast lookup
        self.clinical_features = {}
        for name, path in clinical_feature_paths.items():
            df = pd.read_csv(path).set_index('id')
            self.clinical_features[name] = df
        
        # 3. Create the master list of (patient, task) samples
        self.samples = []
        tasks = [f'task{i}' for i in range(1, 7)] # Assuming tasks are named 'task1', 'task2', etc.
        for _, row in metadata_df.iterrows():
            record_id = row['record_id']
            diagnosis = row['clinical_diagnosis']
            for task_name in tasks:
                self.samples.append((record_id, task_name, diagnosis))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        record_id, task_name, diagnosis = self.samples[idx]
        
        label = self.labels_map[diagnosis]
        
        # --- Prepare data dictionary to be returned ---
        data_dict = {'label': torch.tensor(label, dtype=torch.long)}

        # --- Load Clinical Features if needed ---
        if self.mode in ['clinical', 'fusion']:
            all_task_feats = []
            for df_name in self.clinical_features:
                patient_row = self.clinical_features[df_name].loc[record_id]
                # Select columns for the specific task
                task_cols = [col for col in patient_row.index if col.startswith(f'{task_name}__')]
                all_task_feats.append(patient_row[task_cols].values)
            
            # Concatenate pitch, timing, etc. for the given task
            clinical_tensor = torch.tensor(np.concatenate(all_task_feats), dtype=torch.float32)
            data_dict['clinical_feats'] = clinical_tensor

        # --- Load Embedding Features if needed ---
        if self.mode in ['embedding', 'fusion']:
            # Construct filename, e.g., REDLAT_P001_task1.npz
            npz_path = f"{self.embedding_path}/REDLAT_{record_id}_{task_name}.npz"
            # .npz files contain a default key 'arr_0' if not named
            all_layers = np.load(npz_path)['arr_0'] # Shape: (13, 768)
            
            # Select specified layers and flatten into a single vector
            selected_layers = all_layers[self.embedding_layers, :]
            embedding_tensor = torch.tensor(selected_layers.flatten(), dtype=torch.float32)
            data_dict['embedding_feats'] = embedding_tensor
            
        return data_dict