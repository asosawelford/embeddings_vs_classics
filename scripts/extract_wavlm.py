import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WavLMModel

# --- Configuration ---
# 1. Set the paths for your audio files and where to save embeddings
INPUT_AUDIO_DIR = Path("home/aleph/redlat_mini/tests")
OUTPUT_EMBEDDING_DIR = Path("./wavlm_base_embeddings")

# 2. Set the model name
MODEL_NAME = "microsoft/wavlm-base"

# 3. Choose the device to run the model on
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Main Script ---

def main():
    """
    Main function to extract WavLM embeddings from a folder of audio files.
    """
    print(f"Using device: {DEVICE}")

    # Create the output directory if it doesn't exist
    OUTPUT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    # Load the WavLM processor and model
    print(f"Loading model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval() # Set the model to evaluation mode

    # Find all audio files in the input directory
    # You can add more extensions here if needed (e.g., "*.mp3", "*.flac")
    audio_files = list(INPUT_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")

    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        try:
            # 1. Load and resample the audio file
            # WavLM expects a 16kHz sampling rate
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

            # 2. Process the audio waveform
            # The processor handles normalization and conversion to a tensor
            inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

            # 3. Move inputs to the selected device
            input_values = inputs.input_values.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)

            # 4. Forward pass through the model
            # We don't need to calculate gradients, so we use torch.no_grad()
            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)

            # 5. Get the embeddings
            # The `last_hidden_state` is the sequence of embeddings from the final layer.
            # Its shape is (batch_size, num_frames, hidden_size). Here batch_size is 1.
            last_hidden_state = outputs.last_hidden_state

            # Optional: To get a single fixed-size vector for the entire utterance,
            # you can perform mean pooling over the time dimension.
            # pooled_embedding = torch.mean(last_hidden_state, dim=1)

            # We will save the full sequence of embeddings.
            # Remove the batch dimension and move to CPU before saving as NumPy array.
            embedding_sequence = last_hidden_state.squeeze(0).cpu().numpy()

            # 6. Save the embedding to a file
            # The output filename will be the same as the input but with a .npy extension.
            output_filename = OUTPUT_EMBEDDING_DIR / f"{audio_path.stem}.npy"
            np.save(output_filename, embedding_sequence)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    print("\nExtraction complete.")
    print(f"Embeddings are saved in: {OUTPUT_EMBEDDING_DIR}")

def main_batched():
    """
    Main function to extract WavLM embeddings using batch processing for better performance.
    """
    BATCH_SIZE = 8 # Adjust based on your GPU memory
    print(f"Using device: {DEVICE} with batch size {BATCH_SIZE}")

    OUTPUT_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    audio_files = list(INPUT_AUDIO_DIR.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files to process.")

    # Process files in batches
    for i in tqdm(range(0, len(audio_files), BATCH_SIZE), desc="Extracting embeddings (batched)"):
        batch_paths = audio_files[i:i+BATCH_SIZE]
        batch_audio = []
        
        try:
            for audio_path in batch_paths:
                speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
                batch_audio.append(speech_array)

            # Process the batch of audio waveforms
            # The processor will pad all items in the batch to the same length
            inputs = processor(batch_audio, sampling_rate=16000, return_tensors="pt", padding=True)

            input_values = inputs.input_values.to(DEVICE)
            attention_mask = inputs.attention_mask.to(DEVICE)

            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)

            last_hidden_state = outputs.last_hidden_state.cpu().numpy()

            # The attention mask tells us which frames are padding and which are real data
            # Shape: (batch_size, num_frames)
            mask = attention_mask.cpu().numpy()

            # Save each embedding from the batch
            for j in range(last_hidden_state.shape[0]):
                # Get the actual length of the audio from the attention mask
                actual_length = mask[j].sum()
                
                # Get the corresponding embedding and trim the padding
                embedding = last_hidden_state[j, :actual_length, :]
                
                # Get the original audio path for this item in the batch
                original_audio_path = batch_paths[j]
                
                output_filename = OUTPUT_EMBEDDING_DIR / f"{original_audio_path.stem}.npy"
                np.save(output_filename, embedding)

        except Exception as e:
            print(f"Error processing batch starting with {batch_paths[0]}: {e}")

    print("\nExtraction complete.")
    print(f"Embeddings are saved in: {OUTPUT_EMBEDDING_DIR}")

if __name__ == "__main__":
    # Choose which version to run:
    main()
    # main_batched()
