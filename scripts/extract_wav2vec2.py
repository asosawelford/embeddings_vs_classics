"""
use in cl like this :)

python scripts/extract_wav2vec2.py --input_dir "" --output_dir ""

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
    Main function to extract time-step wav2vec2 embeddings for all layers
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

    # Load the wav2vec2 processor and model
    print(f"Loading model: {args.model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = Wav2Vec2Model.from_pretrained(
        args.model_name,
        output_hidden_states=True # Ensure we get all layers
    ).to(DEVICE)
    model.eval()

    # Find all audio files
    audio_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")
    if not audio_files:
        return

    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Extracting embeddings per time-step"):
        try:
            # --- Preserve directory structure for output file ---
            relative_path = audio_path.relative_to(input_dir)
            output_filename = (output_dir / relative_path).with_suffix('.npz')
            output_filename.parent.mkdir(parents=True, exist_ok=True)

            # 1. Load and resample the audio file
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

            # 2. Process the audio waveform
            inputs = processor(
                speech_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True
            )

            # 3. Ensure tensors have batch dim, create attention_mask if missing, move to device
            input_values = inputs.input_values
            if input_values.dim() == 1:
                input_values = input_values.unsqueeze(0)
            input_values = input_values.to(DEVICE)

            # Create attention_mask if processor didn't return one
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
            else:
                attention_mask = torch.ones_like(input_values, dtype=torch.long)

            attention_mask = attention_mask.to(DEVICE).long()

            # 4. Forward pass through the model
            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)

            all_layer_embeddings = {}
            
            # Get the number of valid (non-padded) frames from the attention mask
            num_valid_frames = int(attention_mask.sum().item())

            for i, layer_hidden_state in enumerate(outputs.hidden_states):
                # Squeeze the batch dimension
                # Shape becomes: (num_frames, hidden_size)
                unbatched_hidden_state = layer_hidden_state.squeeze(0)
                
                # Remove padding by slicing up to the number of valid frames
                unpadded_hidden_state = unbatched_hidden_state[:num_valid_frames, :]
                
                # Move to CPU for saving
                cpu_embedding = unpadded_hidden_state.cpu().numpy()

                all_layer_embeddings[f'layer_{i}'] = cpu_embedding
            
            # Save all layers' time-step embeddings to a single .npz file
            np.savez_compressed(output_filename, **all_layer_embeddings)

        except Exception as e:
            # More informative error logging (print full traceback)
            import traceback
            print(f"\n--- ERROR ---")
            print(f"Failed to process file: {audio_path.name}")
            traceback.print_exc()
            print(f"Skipping this file.")
            print(f"---------------")
            continue  # Move to the next file

    print("\nExtraction complete.")
    print(f"Embeddings are saved in: {output_dir}")



if __name__ == "__main__":
    # --- MODIFIED: Setup argument parser ---
    parser = argparse.ArgumentParser(description="Extract time-step wav2vec2 embeddings from audio files.")
    
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
        help="Name of the wav2vec2 model from the Hugging Face Hub."
    )
    
    args = parser.parse_args()
    main(args)