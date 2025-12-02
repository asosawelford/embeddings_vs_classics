import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
# Directory containing your .pt files
INPUT_DIR = "/home/aleph/redlat/embeddings/processed_word_embeddings.pt" 

# Where to save the report of bad files
REPORT_FILE = "verification_report.csv"

# Expected Tensor Dimensions
EXPECTED_LAYERS = 13
EXPECTED_DIM = 768
# =================================================

def check_file_integrity(filepath):
    """
    Returns a tuple: (is_valid, error_message, num_words)
    """
    try:
        # 1. Try to load the file (CPU is faster/safer for simple checks)
        data = torch.load(filepath, map_location='cpu', weights_only=True)
    except Exception as e:
        return False, f"Load Error: {str(e)}", 0

    # 2. Check for required keys
    required_keys = ['id', 'task', 'embeddings', 'metadata']
    for key in required_keys:
        if key not in data:
            return False, f"Missing Key: {key}", 0

    embeddings = data['embeddings']
    metadata = data['metadata']

    # 3. Check for Empty Data (Valid, but worth noting)
    if len(embeddings) == 0:
        return True, "Warning: Empty (0 words)", 0

    # 4. Check Consistency (Tensor Rows vs Metadata Length)
    if len(embeddings) != len(metadata):
        return False, f"Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata entries", 0

    # 5. Check Tensor Shape
    # Expected: [N_Words, 13, 768]
    if embeddings.dim() != 3:
        return False, f"Wrong Dimensions: Got {embeddings.dim()}D, expected 3D", len(metadata)
    
    if embeddings.shape[1] != EXPECTED_LAYERS:
        return False, f"Wrong Layers: Got {embeddings.shape[1]}, expected {EXPECTED_LAYERS}", len(metadata)
        
    if embeddings.shape[2] != EXPECTED_DIM:
        return False, f"Wrong Dim: Got {embeddings.shape[2]}, expected {EXPECTED_DIM}", len(metadata)

    # 6. Check for NaN/Inf (Corruption check)
    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        return False, "Data Corruption: Contains NaN or Inf values", len(metadata)

    # If we passed all gauntlets
    return True, "OK", len(metadata)

def main():
    input_path = Path(INPUT_DIR)
    files = list(input_path.glob("*.pt"))
    
    if not files:
        print(f"No .pt files found in {INPUT_DIR}")
        return

    print(f"Verifying {len(files)} files in '{INPUT_DIR}'...")
    
    results = []
    corrupt_count = 0
    empty_count = 0
    
    for filepath in tqdm(files):
        is_valid, msg, count = check_file_integrity(filepath)
        
        if not is_valid:
            corrupt_count += 1
            print(f"\n[FAIL] {filepath.name}: {msg}")
            results.append({
                'filename': filepath.name,
                'path': str(filepath),
                'status': 'CORRUPT',
                'details': msg
            })
        elif count == 0:
            empty_count += 1
            # We don't necessarily mark empty files as bad, but good to track
            results.append({
                'filename': filepath.name,
                'path': str(filepath),
                'status': 'EMPTY',
                'details': msg
            })
    
    # Save Report
    if results:
        df = pd.DataFrame(results)
        df.to_csv(REPORT_FILE, index=False)
    
    print("\n" + "="*30)
    print("VERIFICATION SUMMARY")
    print("="*30)
    print(f"Total Files Checked: {len(files)}")
    print(f"Healthy Files:       {len(files) - corrupt_count}")
    print(f"Empty Files:         {empty_count}")
    print(f"Corrupt Files:       {corrupt_count}")
    print("="*30)
    
    if corrupt_count > 0:
        print(f"WARNING: Found {corrupt_count} corrupt files.")
        print(f"Details saved to: {REPORT_FILE}")
        print("Recommendation: Delete these files and re-run the extraction script.")
    else:
        print("All files look perfect!")

if __name__ == "__main__":
    main()