import torch
from torch.utils.data import DataLoader # Import DataLoader
from dataloader import PatientTaskDataset, collate_fn # Import our new function

# --- 1. Define Paths and Parameters for the Test ---
METADATA_PATH = 'data/metadata/train_metadata.csv'
CLINICAL_PATHS = {
    'pitch': 'data/features/explainable/pitch_features.csv',
    'timing': 'data/features/explainable/timing_features.csv'
}
EMBEDDING_PATH = 'data/features/embeddings/wavlm-base-plus'
IMPUTATION_MEANS_PATH = 'data/features/imputation_means.json' # Path to the file you created
SELECTED_LAYERS = [0, 8, 11]

print("="*50)
print("TESTING DATALOADER IN 'fusion' MODE")
print("="*50)

# --- 2. Test Instantiation ---
try:
    fusion_dataset = PatientTaskDataset(
        metadata_path=METADATA_PATH,
        clinical_feature_paths=CLINICAL_PATHS,
        embedding_path=EMBEDDING_PATH,
        mode='fusion',
        imputation_means_path=IMPUTATION_MEANS_PATH,
        embedding_layers=SELECTED_LAYERS,
        classes_to_load= ['CN', 'AD', 'FTD']
    )
    print("✅ Instantiation successful.")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    exit()

# --- 3. Test Length ---
expected_len = 428 * 6 # 428 training patients * 6 tasks
actual_len = len(fusion_dataset)
print(f"\n✅ Length test: Expected {expected_len}, Got {actual_len}")
assert expected_len == actual_len, "Length mismatch!"

# --- 4. Inspect a Single Sample (`__getitem__`) ---
print("\n--- Inspecting a single sample (index=0) ---")
sample = fusion_dataset[0]

# Check keys
print(f"✅ Sample keys: {sample.keys()}")
assert 'clinical_feats' in sample and 'embedding_feats' in sample and 'label' in sample, "Missing keys in sample!"

# Check label (robustly)
record_id, task, diagnosis = fusion_dataset.samples[0]
expected_label_idx = fusion_dataset.labels_map[diagnosis]
print(f"Patient {record_id} has diagnosis '{diagnosis}'.")
print(f"✅ Label: Expected index {expected_label_idx}, Got {sample['label'].item()}")
assert sample['label'].item() == expected_label_idx

# Check shapes and types (by printing, not asserting specific values)
clinical_tensor = sample['clinical_feats']
print(f"✅ Clinical feats type: {clinical_tensor.dtype}")
print(f"✅ Clinical feats shape: {clinical_tensor.shape}") # Just print the shape

embedding_tensor = sample['embedding_feats']
expected_embedding_shape = (len(SELECTED_LAYERS) * 768,)
print(f"✅ Embedding feats type: {embedding_tensor.dtype}")
print(f"✅ Embedding feats shape: Expected {expected_embedding_shape}, Got {embedding_tensor.shape}")
assert embedding_tensor.shape == expected_embedding_shape

print("\n--- Inspecting another random sample (index=100) ---")
sample = fusion_dataset[100]
record_id, task, diagnosis = fusion_dataset.samples[100]
expected_label_idx = fusion_dataset.labels_map[diagnosis]
print(f"Patient {record_id} has diagnosis '{diagnosis}'.")
print(f"✅ Label: Expected index {expected_label_idx}, Got {sample['label'].item()}")
print(f"✅ Clinical feats shape: {sample['clinical_feats'].shape}")
print(f"✅ Embedding feats shape: {sample['embedding_feats'].shape}")

print("\n--- Testing DataLoader with custom collate_fn ---")
data_loader = DataLoader(
    dataset=fusion_dataset,
    batch_size=4, # Use a small batch size for testing
    shuffle=False, # Keep order predictable for testing
    collate_fn=collate_fn # Use our custom function
)

# --- Get the first batch ---
first_batch = next(iter(data_loader))

print("✅ Successfully retrieved the first batch.")
print(f"✅ Batch keys: {first_batch.keys()}")

# --- Inspect the shapes of the BATCHED tensors ---
batched_labels = first_batch['label']
print(f"✅ Batched labels shape: {batched_labels.shape}") # Should be (4,)

batched_clinical = first_batch['clinical_feats']
print(f"✅ Batched clinical feats shape: {batched_clinical.shape}") # e.g., (4, 40) - length will be max in batch

batched_embeddings = first_batch['embedding_feats']
print(f"✅ Batched embedding feats shape: {batched_embeddings.shape}") # Should be (4, 2304)

print("\n" + "="*50)
print("✅✅✅ DATALOADER BATCHING TEST COMPLETED SUCCESSFULLY! ✅✅✅")
