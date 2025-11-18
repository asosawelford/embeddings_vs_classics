import os
import pandas as pd
import numpy as np
import ast
import torch
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_CSV = "path/to/your_transcripts.csv"
NPZ_ROOT_DIR = "path/to/your_npz_folder"
OUTPUT_FILE = "processed_word_embeddings.pt"

# WavLM standard: 1 frame = 20ms (50Hz)
FRAME_RATE = 50 
NUM_LAYERS = 13
DIMENSION = 768
# =================================================

def index_files(root_dir):
    """
    Recursively finds all .npz files and creates a mapping 
    from filename to full path for O(1) access.
    """
    print(f"Indexing files in {root_dir}...")
    file_map = {}
    for path in Path(root_dir).rglob('*.npz'):
        file_map[path.name] = str(path)
    print(f"Found {len(file_map)} .npz files.")
    return file_map

def process_audio_files():
    # 1. Load CSV
    df = pd.read_csv(INPUT_CSV)
    
    # 2. Index the NPZ files
    file_map = index_files(NPZ_ROOT_DIR)
    
    dataset = []
    missing_files = []

    print("Processing records...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        record_id = row['id']
        task = row['task']
        
        # Reconstruct filename based on your naming convention
        # Example: REDLAT_AF022_Semantic.npz
        filename = f"REDLAT_{record_id}_{task}.npz"
        
        if filename not in file_map:
            missing_files.append(filename)
            continue
            
        file_path = file_map[filename]
        
        # 3. Load the NPZ file
        try:
            # Returns a dictionary-like object with keys 'layer_0', 'layer_1', etc.
            npz_data = np.load(file_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # 4. Parse Timestamps
        try:
            # Convert string "['word', 0.1, 0.5]" to list
            timestamps = ast.literal_eval(row['lemmatized_timestamps'])
        except:
            print(f"Error parsing timestamps for {record_id}")
            continue
            
        if not timestamps:
            continue

        # 5. Pre-load all layers into a single tensor for speed
        # Shape: [13, Total_Frames, 768]
        # Note: We assume all layers have the same time length
        try:
            # specific implementation depends on if layers are stacked or separate keys
            # Assuming separate keys based on your description
            max_time = npz_data['layer_0'].shape[0]
            full_audio_tensor = np.zeros((NUM_LAYERS, max_time, DIMENSION), dtype=np.float32)
            
            for i in range(NUM_LAYERS):
                full_audio_tensor[i] = npz_data[f'layer_{i}']
        except KeyError:
            print(f"Layer keys missing in {filename}")
            continue

        # 6. Extract Word Embeddings
        # We want a list of tensors: Shape [13, 768] per word
        word_embeddings_list = []
        word_tokens = []
        
        for word_info in timestamps:
            word_text = word_info[0]
            start_t = float(word_info[1])
            end_t = float(word_info[2])
            
            start_idx = int(start_t * FRAME_RATE)
            end_idx = int(end_t * FRAME_RATE)
            
            # Boundary checks
            if start_idx >= max_time: 
                continue
            if end_idx > max_time: 
                end_idx = max_time
            if start_idx >= end_idx: 
                continue # Skip zero-length words
            
            # SLICE AND AVERAGE
            # Slice: [13, Duration, 768]
            segment = full_audio_tensor[:, start_idx:end_idx, :]
            
            # Mean over the Time dimension (axis 1)
            # Result: [13, 768]
            avg_embedding = np.mean(segment, axis=1)
            
            word_embeddings_list.append(avg_embedding)
            word_tokens.append(word_text)
            
        # 7. Store Result
        if len(word_embeddings_list) > 0:
            # Stack to create [Num_Words, 13, 768]
            final_tensor = torch.tensor(np.stack(word_embeddings_list), dtype=torch.float32)
            
            dataset.append({
                'id': record_id,
                'task': task,
                'embeddings': final_tensor, # The crucial 3D tensor
                'words': word_tokens
            })

    # 8. Save to disk
    torch.save(dataset, OUTPUT_FILE)
    print(f"\nSuccess! Processed {len(dataset)} files.")
    print(f"Saved to {OUTPUT_FILE}")
    if missing_files:
        print(f"Warning: {len(missing_files)} files were not found (e.g., {missing_files[0]})")

if __name__ == "__main__":
    process_audio_files()