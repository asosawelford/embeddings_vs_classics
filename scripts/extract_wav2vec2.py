"""
use in cl like this :)

python scripts/extract_wavlm.py --input_dir "" --output_dir ""

"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def main(args):
    """
    Main function to extract time-step WavLM embeddings for all layers
    from a folder of audio files.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Use paths from arguments
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load the WavLM processor and model
    print(f"Loading model: {args.model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(
        args.model_name,
        output_hidden_states=True # Ensure we get all layers
    ).to(DEVICE)
    model.eval()

    # Find all audio files
    audio_files = list(input_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")
    if not audio_files:
        return

    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Extracting embeddings per time-step"):
        try:
            # 1. Load and resample the audio file
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

            # 2. Process the audio waveform
            inputs = processor(
                speech_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True # Padding is necessary for handling variable lengths
            )

            # 3. Move inputs to the selected device
            input_values = inputs.input_values.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)

            # 4. Forward pass through the model
            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)

            all_layer_embeddings = {}
            
            # Get the number of valid (non-padded) frames from the attention mask
            num_valid_frames = attention_mask.sum().item()

            for i, layer_hidden_state in enumerate(outputs.hidden_states):
                # Squeeze the batch dimension
                # Shape becomes: (num_frames, hidden_size)
                unbatched_hidden_state = layer_hidden_state.squeeze(0)
                
                # Remove padding by slicing up to the number of valid frames
                unpadded_hidden_state = unbatched_hidden_state[:num_valid_frames, :]
                
                # Move to CPU for saving
                cpu_embedding = unpadded_hidden_state.cpu().numpy()

                all_layer_embeddings[f'layer_{i}'] = cpu_embedding
            
            # --- MODIFIED: Save all layers' time-step embeddings to a single .npz file ---
            output_filename = output_dir / f"{audio_path.stem}.npz"
            np.savez_compressed(output_filename, **all_layer_embeddings)

        except Exception as e:
            # More informative error logging
            print(f"\n--- ERROR ---")
            print(f"Failed to process file: {audio_path.name}")
            print(f"Error: {e}")
            print(f"Skipping this file.")
            print(f"---------------")
            continue  # Move to the next file

    print("\nExtraction complete.")
    print(f"Embeddings are saved in: {output_dir}")


if __name__ == "__main__":
    # --- MODIFIED: Setup argument parser ---
    parser = argparse.ArgumentParser(description="Extract time-step WavLM embeddings from audio files.")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing .wav audio files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where embeddings will be saved as .npz files."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="microsoft/wavlm-base-plus",
        default="facebook/wav2vec2-base",
        help="Name of the WavLM model from the Hugging Face Hub."
    )
    
    args = parser.parse_args()
    main(args)