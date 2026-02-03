import torch
import pandas as pd
import functools
from torch.utils.data import DataLoader
from dataloader import PatientTaskDataset, collate_fn

# --- 1. Define Paths and Parameters for the Test ---
METADATA_PATH = 'data/metadata_lopera.csv'
# Clinical paths are kept to ensure the dataloader ignores them correctly in 'embedding' mode
CLINICAL_PATHS = {
    'pitch': 'data/features/explainable/pitch_features.csv',
    'timing': 'data/features/explainable/timing_features.csv'
}
EMBEDDING_PATH = 'data/features/embeddings/wavlm-base-plus'
IMPUTATION_MEANS_PATH = 'data/features/imputation_means.json'
SCALING_PARAMS_PATH = "data/features/scaling_params.json"
SELECTED_LAYERS = [0, 8, 11]
CLASSES_TO_LOAD = ['CN', 'AD'] # Dataloader is designed for binary tasks
BATCH_SIZE = 4

# --- Dynamically calculate expected length ---
try:
    meta_df = pd.read_csv(METADATA_PATH)
    num_patients = meta_df[meta_df['clinical_diagnosis'].isin(CLASSES_TO_LOAD)].shape[0]
    EXPECTED_LEN = num_patients * 6 # num_patients * num_tasks
except FileNotFoundError:
    print(f"❌ Warning: Metadata file not found at {METADATA_PATH}. Cannot calculate expected length.")
    EXPECTED_LEN = 0 # Set to 0 to let the test run and report mismatch

print("="*50)
print("TESTING DATALOADER IN 'embedding' MODE")
print("="*50)

# --- 2. Test Instantiation ---
try:
    embedding_dataset = PatientTaskDataset(
        metadata_path=METADATA_PATH,
        clinical_feature_paths=CLINICAL_PATHS, # These should be ignored
        embedding_path=EMBEDDING_PATH,
        mode='embedding', # <<--- The key change is here
        imputation_means_path=IMPUTATION_MEANS_PATH, # Should be ignored
        scaling_params_path=SCALING_PARAMS_PATH, # Should be ignored
        embedding_layers=SELECTED_LAYERS,
        classes_to_load=CLASSES_TO_LOAD
    )
    print("✅ Instantiation successful.")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    exit()

# --- 3. Test Length ---
actual_len = len(embedding_dataset)
print(f"\n✅ Length test: Expected {EXPECTED_LEN}, Got {actual_len}")
if EXPECTED_LEN > 0:
    assert EXPECTED_LEN == actual_len, "Length mismatch!"

# --- 4. Inspect a Single Sample (`__getitem__`) ---
print("\n--- Inspecting a single sample (index=0) ---")
sample = embedding_dataset[0]

# Check keys
print(f"✅ Sample keys: {sample.keys()}")
assert 'embedding_feats' in sample, "Missing 'embedding_feats' key in sample!"
assert 'label' in sample, "Missing 'label' key in sample!"
assert 'clinical_feats' not in sample, "Found 'clinical_feats', which should NOT be present in embedding mode!"
print("✅ Key check successful.")

# Check label (robustly)
record_id, task, diagnosis = embedding_dataset.samples[0]
expected_label_idx = embedding_dataset.labels_map[diagnosis]
print(f"Patient {record_id} has diagnosis '{diagnosis}'.")
print(f"✅ Label: Expected index {expected_label_idx}, Got {sample['label'].item()}")
assert sample['label'].item() == expected_label_idx

# Check shapes and types for embedding features
embedding_tensor = sample['embedding_feats']
expected_embedding_shape = (len(SELECTED_LAYERS) * 768,)
print(f"✅ Embedding feats type: {embedding_tensor.dtype}")
print(f"✅ Embedding feats shape: Expected {expected_embedding_shape}, Got {embedding_tensor.shape}")
assert embedding_tensor.shape == expected_embedding_shape

print("\n--- Testing DataLoader with custom collate_fn ---")
# In embedding mode, fixed_clinical_len is not needed, but we can create the partial function
# to be consistent with the main training script.
collate_function = functools.partial(collate_fn, fixed_clinical_len=None)

data_loader = DataLoader(
    dataset=embedding_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_function
)

# --- Get the first batch ---
first_batch = next(iter(data_loader))

print("\n✅ Successfully retrieved the first batch.")
print(f"✅ Batch keys: {first_batch.keys()}")
assert 'clinical_feats' not in first_batch, "Found 'clinical_feats' in batch, which should not be present!"
print("✅ Batch key check successful.")


# --- Inspect the shapes of the BATCHED tensors ---
batched_labels = first_batch['label']
print(f"✅ Batched labels shape: {batched_labels.shape}")
assert batched_labels.shape == (BATCH_SIZE,)

batched_embeddings = first_batch['embedding_feats']
print(f"✅ Batched embedding feats shape: {batched_embeddings.shape}")
assert batched_embeddings.shape == (BATCH_SIZE, len(SELECTED_LAYERS) * 768)

print("\n" + "="*50)
print("✅✅✅ EMBEDDING MODE DATALOADER TEST COMPLETED SUCCESSFULLY! ✅✅✅")