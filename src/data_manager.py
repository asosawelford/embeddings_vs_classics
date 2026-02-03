import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

class AlzheimerDataset(Dataset):
    def __init__(self, features, labels, record_ids, tasks):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.record_ids = record_ids
        self.tasks = tasks

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'record_id': self.record_ids[idx],
            'task': self.tasks[idx]
        }

class DataManager:
    def __init__(self, metadata_path, target_groups=['AD', 'CN']):
        self.target_groups = target_groups
        
        # Load metadata
        df = pd.read_csv(metadata_path, low_memory=False)
        
        # Filter groups
        df = df[df['clinical_diagnosis'].isin(target_groups)].copy()
        self.metadata = df.reset_index(drop=True)
        
        # Encode labels
        self.le = LabelEncoder()
        self.metadata['label_encoded'] = self.le.fit_transform(self.metadata['clinical_diagnosis'])
        
        # Get unique patients for splitting
        self.unique_patients = self.metadata['record_id'].unique()
        
        print(f"--- DataManager Initialized ---")
        print(f"Groups: {target_groups}")
        print(f"Patients: {len(self.unique_patients)}")
        print(f"Class Map: {dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))}")

    def get_labels_for_splitting(self):
        # Helper to get one label per patient for StratifiedKFold
        patient_labels = []
        for pid in self.unique_patients:
            lbl = self.metadata[self.metadata['record_id'] == pid]['label_encoded'].iloc[0]
            patient_labels.append(lbl)
        return np.array(patient_labels)

    def split_patients(self, train_patient_ids, test_patient_ids):
        train_df = self.metadata[self.metadata['record_id'].isin(train_patient_ids)]
        test_df = self.metadata[self.metadata['record_id'].isin(test_patient_ids)]
        return train_df, test_df

    def load_embeddings(self, df, embedding_dir, tasks, model_type='wavlm'):
        X = []
        y = []
        ids = []
        task_list = []
        
        missing_count = 0
        embedding_dir = Path(embedding_dir)
        printed_debug = False

        for _, row in df.iterrows():
            record_id = row['record_id']
            site = row['site']
            label = row['label_encoded']
            
            for task_name in tasks:
                # Path Logic: Base / Site / ID / REDLAT_ID_Task.npz
                filename = f"REDLAT_{record_id}_{task_name}.npz"
                file_path = embedding_dir / site / record_id / filename

                if not file_path.exists():
                    if not printed_debug:
                        print(f"\n❌ [DEBUG PATH CHECK]")
                        print(f"Looking for: {file_path}")
                        print(f"Result: FILE NOT FOUND (Skipping others)")
                        printed_debug = True
                    missing_count += 1
                    continue

                try:
                    data = np.load(file_path)
                    
                    if model_type == 'wavlm':
                        if 'embeddings' in data:
                            emb = data['embeddings']
                        else:
                            emb = np.stack([data[f'layer_{i}'] for i in range(13)])
                            
                    elif model_type == 'roberta':
                        if 'embedding' in data:
                            emb = data['embedding']
                        else:
                            emb = data['embeddings']

                    X.append(emb)
                    y.append(label)
                    ids.append(record_id)
                    task_list.append(task_name)
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        if missing_count > 0:
            print(f"⚠️  Skipped {missing_count} missing files.")

        return np.array(X), np.array(y), ids, task_list

    def load_classic_features(self, train_df, test_df, csv_path, tasks, k=5, subset=None):
        """
        subset (str): 'audio', 'language', or None (all)
        """
        feat_df = pd.read_csv(csv_path, low_memory=False)
        
        # --- DEFINITIONS OF FEATURE FAMILIES ---
        # Edit these lists to match the substrings in your column headers
        FEATURE_FAMILIES = {
            'audio': [
                'pitch_analysis_pitch', 
                'talking_intervals',
            ],
            'language': [
                'concreteness', 
                'granularity', 
                'verbosity', 
                'OSV',
                'psycholinguistic_objective'
            ]
        }
        
        def _expand_classic(meta_df, is_train, imputer=None, scaler=None, k_neighbors=5):
            # 1. Lean Metadata
            lean_meta = meta_df[['record_id', 'label_encoded']].copy()
            merged = pd.merge(lean_meta, feat_df, on='record_id', how='inner')
            
            # 2. TASK & SUBSET FILTERING
            if 'task' in merged.columns:
                # Long Format Logic (Rows)
                merged = merged[merged['task'].isin(tasks)]
                # (Subset filtering for Long format would happen later on columns, 
                # but let's assume Wide format as per your files)
                t_names = merged['task'].values
            else:
                # Wide Format Logic (Columns)
                t_names = ['combined'] * len(merged)
                cols_to_keep = ['record_id', 'label_encoded']
                feature_cols = []
                
                # Get keywords for the requested subset
                keywords = FEATURE_FAMILIES.get(subset) if subset else None

                for col in merged.columns:
                    if col in cols_to_keep: continue
                    
                    # 1. Check Task (Starts with...)
                    is_task_match = False
                    for t in tasks:
                        if col.startswith(t):
                            is_task_match = True
                            break
                    if not is_task_match: continue
                    
                    # 2. Check Subset (Contains string...)
                    if keywords:
                        # Only keep if column name contains at least one keyword
                        if not any(k in col for k in keywords):
                            continue 
                            
                    feature_cols.append(col)
                
                merged = merged[cols_to_keep + feature_cols]

            # 3. Anti-Leakage Drop
            cols_to_drop = ['record_id', 'label_encoded', 'task', 'site', 'criteria_category', 'clinical_diagnosis']
            cols_to_drop += ['participant_id', 'subject', 'ID', 'Unnamed: 0']
            existing_drop = [c for c in cols_to_drop if c in merged.columns]
            
            features_df = merged.drop(columns=existing_drop)
            
            # 4. Extract
            features = features_df.select_dtypes(include=[np.number]).values
            labels = merged['label_encoded'].values
            r_ids = merged['record_id'].values

            if features.shape[1] == 0:
                raise ValueError(f"No features found for Task={tasks} and Subset={subset}. Check your keywords!")

            if is_train:
                imputer = KNNImputer(n_neighbors=k_neighbors)
                features_imp = imputer.fit_transform(features)
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_imp)
                return features_scaled, labels, r_ids, t_names, imputer, scaler
            else:
                features_imp = imputer.transform(features)
                features_scaled = scaler.transform(features_imp)
                return features_scaled, labels, r_ids, t_names, None, None

        X_train, y_train, id_train, t_train, imp, scl = _expand_classic(train_df, True, k_neighbors=k)
        X_test, y_test, id_test, t_test, _, _ = _expand_classic(test_df, False, imputer=imp, scaler=scl)
        
        return (X_train, y_train, id_train, t_train), (X_test, y_test, id_test, t_test)