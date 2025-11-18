"""
This script takes a directory of WavLM embeddings (saved as .npz files)
and applies time-pooling to each layer.

It reads .npz files where each file contains multiple arrays ('layer_0', 'layer_1', etc.),
each with shape (time_steps, 768). It computes the mean across the time_steps
for each layer and saves the result as a single (13, 768) array in a .npy file.

Use in CLI like this:
python pool_embeddings.py --input_dir "path/to/unpooled_embeddings" --output_dir "path/to/pooled_embeddings"
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def main(args):
    """
    Main function to process .npz embeddings, apply time-pooling,
    and save the results as .npy files.
    """
    # Use paths from arguments
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # --- Recursively find all .npz files in subdirectories ---
    npz_files = list(input_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files to process.")
    if not npz_files:
        return

    # Process each .npz file
    for npz_path in tqdm(npz_files, desc="Pooling embeddings"):
        try:
            # --- Preserve directory structure for output file ---
            # 1. Get the relative path of the .npz file to the input directory
            relative_path = npz_path.relative_to(input_dir)

            # 2. Create the full output path, changing the extension to .npy
            output_filename = (output_dir / relative_path).with_suffix('.npy')

            # 3. Create parent directories for the output file if they don't exist
            output_filename.parent.mkdir(parents=True, exist_ok=True)

            # --- Load, Pool, and Stack ---
            # 1. Load the .npz file
            # np.load returns a LazyLoader that acts like a dictionary
            with np.load(npz_path) as data:
                all_pooled_layers = []
                
                # The WavLM-Base+ model has 13 layers (1 input + 12 transformer blocks)
                # The keys are 'layer_0', 'layer_1', ..., 'layer_12'
                num_layers = len(data.files)
                if num_layers != 13:
                    print(f"\nWarning: Expected 13 layers but found {num_layers} in {npz_path.name}. Processing what's available.")

                for i in range(num_layers):
                    layer_key = f'layer_{i}'
                    # a. Get the (time, 768) embedding for the current layer
                    time_series_embedding = data[layer_key]
                    
                    # b. Perform time-pooling by taking the mean across the time axis (axis=0)
                    # This reduces (time, 768) -> (768,)
                    pooled_vector = np.mean(time_series_embedding, axis=0)
                    all_pooled_layers.append(pooled_vector)

                # 2. Stack the list of pooled vectors into a single array
                # This creates a (13, 768) numpy array
                final_array = np.stack(all_pooled_layers, axis=0)

            # 3. Save the final (13, 768) array to a .npy file
            np.save(output_filename, final_array)

        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"Failed to process file: {npz_path.name}")
            print(f"Error: {e}")
            print(f"Skipping this file.")
            print(f"---------------")
            continue

    print("\nPooling complete.")
    print(f"Pooled embeddings are saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pool time-step .npz embeddings into fixed-size .npy files."
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root directory containing .npz embedding files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where pooled .npy embeddings will be saved."
    )
    
    args = parser.parse_args()
    main(args)