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
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

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
    model = WavLMModel.from_pretrained(
        args.model_name,
        output_hidden_states=True # Ensure we get all layers
    ).to(DEVICE)
    model.eval()

    # --- Recursively find all audio files in subdirectories ---
    audio_files = list(input_dir.rglob("*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")
    if not audio_files:
        return

    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Extracting embeddings per time-step"):
        try:
            # --- Preserve directory structure for output file ---
            # 1. Get the path of the audio file relative to the input directory
            #    e.g., if input_dir is 'data/' and audio_path is 'data/speaker1/file.wav',
            #    relative_path will be 'speaker1/file.wav'
            relative_path = audio_path.relative_to(input_dir)

            # 2. Create the full output path by joining the output_dir and the relative_path,
            #    and then change the file extension from .wav to .npz
            output_filename = (output_dir / relative_path).with_suffix('.npz')

            # 3. Create the parent directories for the output file if they don't exist
            #    e.g., create 'output_dir/speaker1/'
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

            # 3. Move inputs to the selected device
            input_values = inputs.input_values.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)

            # 4. Forward pass through the model
            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)

            # --- Keep all time-step embeddings for each layer ---
            all_layer_embeddings = {}
            num_valid_frames = attention_mask.sum().item()

            for i, layer_hidden_state in enumerate(outputs.hidden_states):
                unbatched_hidden_state = layer_hidden_state.squeeze(0)
                unpadded_hidden_state = unbatched_hidden_state[:num_valid_frames, :]
                cpu_embedding = unpadded_hidden_state.cpu().numpy()
                all_layer_embeddings[f'layer_{i}'] = cpu_embedding
            
            # --- Save all layers' time-step embeddings to the new path ---
            np.savez_compressed(output_filename, **all_layer_embeddings)

        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"Failed to process file: {audio_path.name}")
            print(f"Error: {e}")
            print(f"Skipping this file.")
            print(f"---------------")
            continue

    print("\nExtraction complete.")
    print(f"Embeddings are saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract time-step WavLM embeddings from all audio files in a directory and its subdirectories."
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory containing .wav audio files (subdirectories will be searched)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where embeddings will be saved, preserving the input folder structure."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/wavlm-base-plus",
        help="Name of the WavLM model from the Hugging Face Hub."
    )
    
    args = parser.parse_args()
    main(args)