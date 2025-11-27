import pandas as pd
import ast
import torch
import librosa
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# ================= CONSTANTS =================
# WavLM operates at 50Hz (1 frame = 20ms)
FRAME_RATE = 50 
# =============================================

def index_audio_files(input_dir):
    """
    Recursively finds all .wav files and creates a mapping 
    from filename to full path for O(1) access.
    """
    print(f"Indexing audio files in {input_dir}...")
    file_map = {}
    # We index by the filename (e.g., 'REDLAT_AF001_reading.wav')
    for path in Path(input_dir).rglob('*.wav'):
        file_map[path.name] = str(path)
    print(f"Found {len(file_map)} .wav files.")
    return file_map

def main(args):
    # 1. Setup Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 2. Load Model & Processor
    print(f"Loading WavLM: {args.model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = WavLMModel.from_pretrained(
        args.model_name, 
        output_hidden_states=True
    ).to(DEVICE)
    model.eval()

    # 3. Load the CSV Data
    # This is the CSV we created in Step 1
    df = pd.read_csv(args.csv_path)
    print(f"Loaded metadata CSV with {len(df)} rows.")

    # 4. Index Audio Files
    file_map = index_audio_files(args.audio_dir)
    
    # 5. Prepare Output Directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Starting One-Pass Extraction & Alignment ---")
    
    # Iterate through the dataframe
    for _, row in tqdm(df.iterrows(), total=len(df)):
        record_id = row['id']
        task = row['task']
        
        # Construct Output Filename
        output_filename = output_dir / f"{record_id}_{task}_embeddings.pt"
        
        # SKIP if already exists (Resume capability)
        if output_filename.exists():
            continue

        # Construct Input Filename
        # CHANGE THIS format if your audio files are named differently
        # Example assumption: REDLAT_AF001_reading.wav
        wav_filename = f"REDLAT_{record_id}_{task}.wav"
        
        if wav_filename not in file_map:
            # Optional: Try a fallback naming convention if needed
            # print(f"Warning: Audio {wav_filename} not found.")
            continue
            
        audio_path = file_map[wav_filename]

        # Parse the Content Word Details (The list of dicts)
        try:
            word_details = ast.literal_eval(row['content_word_details'])
        except:
            print(f"Error parsing metadata for {record_id}")
            continue
            
        if not word_details:
            continue

        try:
            # --- A. LOAD AUDIO ---
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

            # --- B. RUN WAVLM INFERENCE ---
            inputs = processor(
                speech_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            input_values = inputs.input_values.to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_values)

            # Stack all layers: Shape [13, 1, Time, 768] -> [13, Time, 768]
            # WavLM Base+ has 13 layers (0 to 12)
            hidden_states = torch.stack(outputs.hidden_states).squeeze(1)
            
            # Dimensions
            num_layers, max_frames, dim = hidden_states.shape

            # --- C. SLICE & POOL (The Magic Step) ---
            embeddings_list = []
            metadata_list = []

            for word_info in word_details:
                # word_info looks like: 
                # {'word': 'casa', 'pos': 'NOUN', 'start': 0.5, 'end': 0.9}
                
                start_t = float(word_info['start'])
                end_t = float(word_info['end'])
                
                # Convert time to frame index
                start_idx = int(start_t * FRAME_RATE)
                end_idx = int(end_t * FRAME_RATE)

                # Validations
                if start_idx >= max_frames: continue
                if end_idx > max_frames: end_idx = max_frames
                if start_idx >= end_idx: continue # Duration is 0

                # Slice the tensor on GPU: [13, Duration, 768]
                segment = hidden_states[:, start_idx:end_idx, :]

                # Average over time dimension (dim=1): Result [13, 768]
                # We keep it as a tensor for now
                pooled_embedding = torch.mean(segment, dim=1)
                
                # Move to CPU to save memory (we are done with GPU for this word)
                embeddings_list.append(pooled_embedding.cpu())
                
                metadata_list.append({
                    'word': word_info['word'],
                    'lemma': word_info.get('lemma', ''),
                    'pos': word_info['pos']
                })

            # --- D. SAVE TO DISK ---
            if embeddings_list:
                # Stack into one tensor: [Num_Words, 13, 768]
                final_tensor = torch.stack(embeddings_list)
                
                save_payload = {
                    'id': record_id,
                    'task': task,
                    'embeddings': final_tensor, # The Tensor
                    'metadata': metadata_list   # The info (Noun, Verb, etc)
                }
                
                torch.save(save_payload, output_filename)
                
            # Clean up GPU memory explicitely (optional but good for long loops)
            del hidden_states
            del outputs
            del input_values
            
        except Exception as e:
            print(f"Error processing {record_id}: {e}")
            continue

    print(f"\nProcessing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WavLM embeddings aligned to content words.")
    
    parser.add_argument("--csv_path", type=str, required=True, 
                        help="Path to lemmatized_content_data.csv")
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Root folder containing .wav files")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Where to save the .pt files")
    parser.add_argument("--model_name", type=str, default="microsoft/wavlm-base-plus", 
                        help="HuggingFace model name")

    args = parser.parse_args()
    main(args)


"""
how to run

python extract_and_align.py \
  --csv_path "lemmatized_content_data.csv" \
  --audio_dir "path/to/your/wavs" \
  --output_dir "final_embeddings"

"""