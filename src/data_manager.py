import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

# --- GLOBAL DEFINITIONS FOR FEATURE FAMILIES ---
# These are used to categorize the curated features into "Audio" or "Text" buckets
AUDIO_KEYS = [
    'pitch_analysis_pitch', 
    'talking_intervals',
    'intensity', 'formant', 'mfcc', 'zero_crossing', 'shimmer', 'jitter', 'HNR'
]
LANG_KEYS = [
    'concreteness', 
    'granularity', 
    'verbosity', 
    'OSV',
    'psycholinguistic_objective',
    'semantic_acuity',
    'graphs',
    'lexical', 
    'syntactic'
]

# --- CURATED FEATURE PATHS ---
# Maps a generic task key to the specific CSV containing the list of relevant features
TASK_FEATURE_FILES = {
    'CraftIm': '/home/aleph/embeddings_vs_classics/data/task_relevant_features/CraftIm_task.csv',
    'Fugu': '/home/aleph/embeddings_vs_classics/data/task_relevant_features/Fugu_task.csv',
    'Phonological': '/home/aleph/embeddings_vs_classics/data/task_relevant_features/Phonological_fluency_task.csv',
    'Semantic': '/home/aleph/embeddings_vs_classics/data/task_relevant_features/Semantic_fluency_task.csv'
}

class AlzheimerDataset(Dataset):
    def __init__(self, features, labels, features_text=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.features_text = torch.FloatTensor(features_text) if features_text is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch = {'features': self.features[idx], 'label': self.labels[idx]}
        if self.features_text is not None:
            batch['features_text'] = self.features_text[idx]
        return batch

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

    def _get_curated_feature_list(self, tasks):
        """
        Reads the CSVs defined in TASK_FEATURE_FILES for the requested tasks.
        Returns a Set of allowed column names.
        """
        allowed_features = set()
        found_config = False

        for task in tasks:
            # We try to match the task string to our keys (e.g., 'Semantic' matches 'Semantic_fluency_task')
            # or exact match
            key_match = None
            for key in TASK_FEATURE_FILES:
                if key in task or task in key:
                    key_match = key
                    break
            
            if key_match:
                path = Path(TASK_FEATURE_FILES[key_match])
                if path.exists():
                    try:
                        f_df = pd.read_csv(path)
                        if 'Feature_name' in f_df.columns:
                            feats = f_df['Feature_name'].dropna().tolist()
                            allowed_features.update(feats)
                            found_config = True
                            # print(f"Loaded {len(feats)} features for task {task}")
                    except Exception as e:
                        print(f"Error reading feature file {path}: {e}")
                else:
                    print(f"Warning: Feature file not found: {path}")

        return allowed_features if found_config else None

    def load_embeddings(self, df, embedding_dir, tasks, model_type='wavlm'):
        X, y, ids, task_list = [], [], [], []
        missing_count = 0
        embedding_dir = Path(embedding_dir)
        printed_debug = False

        for _, row in df.iterrows():
            record_id, site, label = row['record_id'], row['site'], row['label_encoded']
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
                    
                    # Add support for wav2vec2/xlsr (similar structure to WavLM dict)
                    elif model_type == 'xlsr': 
                         # Assumes dictionary with keys 'layer_0'...'layer_24'
                         keys = [k for k in data.keys() if k.startswith('layer_')]
                         # Sort keys to ensure layer order
                         keys.sort(key=lambda x: int(x.split('_')[1]))
                         emb = np.stack([data[k] for k in keys])

                    X.append(emb)
                    y.append(label)
                    ids.append(record_id)
                    task_list.append(task_name)
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        if missing_count > 0:
            print(f"⚠️ Skipped {missing_count} missing files.")

        return np.array(X), np.array(y), ids, task_list

    def load_classic_features(self, train_df, test_df, csv_path, tasks, k=5, subset=None):
        """
        subset (str): 'audio', 'language', or None (all)
        """
        feat_df = pd.read_csv(csv_path, low_memory=False)
        
        # 1. Get Curated List from CSVs
        curated_whitelist = self._get_curated_feature_list(tasks)
        # if curated_whitelist:
        #     print(f"Using Curated Feature List: {len(curated_whitelist)} allowed features found.")

        def _expand_classic(meta_df, is_train, imputer=None, scaler=None, k_neighbors=5):
            # 1. Lean Metadata
            lean_meta = meta_df[['record_id', 'label_encoded']].copy()
            merged = pd.merge(lean_meta, feat_df, on='record_id', how='inner')
            
            # 2. TASK & SUBSET FILTERING
            cols_to_keep = ['record_id', 'label_encoded']
            feature_cols = []
            
            # Define keywords for subset filtering
            if subset == 'audio':
                keywords = AUDIO_KEYS
            elif subset == 'language':
                keywords = LANG_KEYS
            elif subset == 'combined':
                keywords = AUDIO_KEYS + LANG_KEYS
            else:
                keywords = None # Default: Loads all numeric columns

            for col in merged.columns:
                if col in cols_to_keep: continue
                
                # A. Must match one of the requested tasks
                is_task_match = any(t in col for t in tasks) # Looser check: 'Semantic' in 'Semantic_fluency...'
                if not is_task_match: continue

                # B. Must be in Curated Whitelist (if it exists)
                if curated_whitelist is not None:
                    # We check if the column name exists in the whitelist
                    if col not in curated_whitelist:
                        continue

                # C. Must match the Subset (Audio/Language) via Keywords
                if keywords and not any(k in col for k in keywords): continue 
                
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
            
            # If wide format, task names are implicit, just fill with 'combined'
            t_names = ['combined'] * len(merged)

            if features.shape[1] == 0:
                print(f"Warning: No features found for Task={tasks}, Subset={subset}")
                # return empty arrays to avoid crash, or raise error
                # raise ValueError(f"No features found for Task={tasks} and Subset={subset}.")

            if is_train:
                imputer = KNNImputer(n_neighbors=k_neighbors)
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(imputer.fit_transform(features))
                return features_scaled, labels, r_ids, t_names, imputer, scaler
            else:
                return scaler.transform(imputer.transform(features)), labels, r_ids, t_names, None, None

        X_train, y_train, id_train, t_train, imp, scl = _expand_classic(train_df, True, k_neighbors=k)
        X_test, y_test, id_test, t_test, _, _ = _expand_classic(test_df, False, imputer=imp, scaler=scl)
        return (X_train, y_train, id_train, t_train), (X_test, y_test, id_test, t_test)
    
    def load_paired_embeddings(self, df, wavlm_dir, roberta_dir, tasks):
        X_audio, X_text, y = [], [], []
        for _, row in df.iterrows():
            record_id, site, label = row['record_id'], row['site'], row['label_encoded']
            for task in tasks:
                path_wav = Path(wavlm_dir) / site / record_id / f"REDLAT_{record_id}_{task}.npz"
                path_rob = Path(roberta_dir) / site / record_id / f"REDLAT_{record_id}_{task}.npz"
                if not path_wav.exists() or not path_rob.exists(): continue
                
                try:
                    data_w = np.load(path_wav)
                    emb_w = data_w['embeddings'] if 'embeddings' in data_w else np.stack([data_w[f'layer_{i}'] for i in range(13)])
                    data_r = np.load(path_rob)
                    emb_r = data_r['embedding'] if 'embedding' in data_r else data_r['embeddings']
                    
                    X_audio.append(emb_w); X_text.append(emb_r); y.append(label)
                except Exception as e:
                    print(f"Error loading pair for {record_id}: {e}")
        return np.array(X_audio), np.array(X_text), np.array(y)
    
    def load_dual_classic_features(self, train_df, test_df, csv_path, tasks, k=5):
        feat_df = pd.read_csv(csv_path, low_memory=False)

        # 1. Get Curated List
        curated_whitelist = self._get_curated_feature_list(tasks)
        # if curated_whitelist:
        #     print(f"Using Curated Feature List (Dual): {len(curated_whitelist)} allowed features.")

        def _extract_dual(meta_df, is_train, imp_a=None, scl_a=None, imp_l=None, scl_l=None):
            lean_meta = meta_df[['record_id', 'label_encoded']].copy()
            merged = pd.merge(lean_meta, feat_df, on='record_id', how='inner')
            
            # Filter cols based on Task + Whitelist + Modality Key
            cols_a = []
            cols_l = []
            
            for col in merged.columns:
                # Basic task check
                if not any(t in col for t in tasks): continue
                
                # Whitelist check
                if curated_whitelist and col not in curated_whitelist: continue
                
                # Modality check
                if any(k in col for k in AUDIO_KEYS):
                    cols_a.append(col)
                elif any(k in col for k in LANG_KEYS):
                    cols_l.append(col)
            
            print(f"  Selected {len(cols_a)} Audio feats and {len(cols_l)} Text feats.")
            
            X_a = merged[cols_a].select_dtypes(include=[np.number]).values
            X_l = merged[cols_l].select_dtypes(include=[np.number]).values
            labels = merged['label_encoded'].values

            if is_train:
                imp_a, scl_a = KNNImputer(n_neighbors=k), StandardScaler()
                X_a = scl_a.fit_transform(imp_a.fit_transform(X_a))
                imp_l, scl_l = KNNImputer(n_neighbors=k), StandardScaler()
                X_l = scl_l.fit_transform(imp_l.fit_transform(X_l))
                return X_a, X_l, labels, imp_a, scl_a, imp_l, scl_l
            else:
                return scl_a.transform(imp_a.transform(X_a)), scl_l.transform(imp_l.transform(X_l)), labels, None, None, None, None

        X_a_tr, X_l_tr, y_tr, ia, sa, il, sl = _extract_dual(train_df, True)
        X_a_te, X_l_te, y_te, _, _, _, _ = _extract_dual(test_df, False, ia, sa, il, sl)
        return (X_a_tr, X_l_tr, y_tr), (X_a_te, X_l_te, y_te)