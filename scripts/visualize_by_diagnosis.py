import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re
from tqdm import tqdm
import umap

def main(args):
    """
    Loads embeddings and clinical data, performs UMAP, and creates a 2D scatter plot
    colored by clinical diagnosis.
    """
    embedding_dir = Path(args.embedding_dir)
    metadata_file = Path(args.metadata_file)
    layer_key = f'layer_{args.layer}'
    
    print(f"Analyzing embeddings from: {embedding_dir}")
    print(f"Loading metadata from: {metadata_file}")
    print(f"Using layer: {layer_key}")
    print(f"Coloring by column: '{args.color_by_column}'")

    # --- 1. Load Metadata from Excel File ---
    try:
        df_meta = pd.read_excel(metadata_file)
        # Set 'record_id' as the index for fast lookups
        df_meta.set_index('record_id', inplace=True)
        print(f"Successfully loaded metadata for {len(df_meta)} records.")
    except Exception as e:
        print(f"Error: Could not load or process metadata file '{metadata_file}'.")
        print(f"Details: {e}")
        return

    # --- 2. Load Embeddings and Match with Metadata ---
    all_embeddings = []
    diagnosis_labels = []
    
    # This regex now extracts the full record_id (e.g., "AF022")
    record_id_pattern = re.compile(r"_([A-Z]{2}\d+)_")

    npz_files = list(embedding_dir.glob("*.npz"))
    if not npz_files:
        print(f"Error: No .npz files found in {embedding_dir}")
        return

    print("\nLoading embeddings and matching with clinical diagnosis...")
    for file_path in tqdm(npz_files, desc="Processing files"):
        match = record_id_pattern.search(file_path.name)
        
        if match:
            record_id = match.group(1) # e.g., "AF022"
            
            # Look up the diagnosis in our DataFrame
            if record_id in df_meta.index:
                diagnosis = df_meta.loc[record_id, args.color_by_column]
                
                # Handle cases where diagnosis might be missing (NaN)
                if pd.isna(diagnosis):
                    print(f"Warning: Missing diagnosis for record '{record_id}' in column '{args.color_by_column}'. Skipping.")
                    continue
                
                # Load the corresponding embedding
                try:
                    with np.load(file_path) as data:
                        if layer_key in data:
                            all_embeddings.append(data[layer_key])
                            diagnosis_labels.append(str(diagnosis)) # Ensure label is a string
                        else:
                            print(f"Warning: Layer '{layer_key}' not found in {file_path.name}. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not load embedding {file_path.name}. Error: {e}. Skipping.")
            else:
                print(f"Warning: Record ID '{record_id}' from filename not found in metadata. Skipping.")
        else:
            print(f"Warning: Record ID pattern not found in filename '{file_path.name}'. Skipping.")

    if not all_embeddings:
        print("\nError: No valid embeddings could be matched with metadata. Exiting.")
        return

    # Convert lists to NumPy arrays
    X = np.array(all_embeddings)
    y_labels = np.array(diagnosis_labels)
    
    print(f"\nSuccessfully loaded and matched {len(X)} samples.")
    unique_diagnoses = np.unique(y_labels)
    print(f"Found diagnoses: {unique_diagnoses}")

    # --- 3. Reduce Dimensionality with UMAP ---
    print("\nPerforming UMAP to reduce from 768 to 2 dimensions...")
    # You can tune these parameters, but defaults are a good start
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    # --- 4. Visualize the Results ---
    print("Creating scatter plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 10))

    # Using seaborn for better aesthetics and automatic legend handling
    # Choose a palette that has enough distinct colors
    palette = sns.color_palette("tab10", n_colors=len(unique_diagnoses))

    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=y_labels,
        hue_order=sorted(unique_diagnoses), # Keep legend order consistent
        palette=palette,
        alpha=0.8,
        s=60
    )

    plt.title(f'UMAP of WavLM Layer {args.layer} Embeddings by {args.color_by_column}', fontsize=18)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(title=args.color_by_column.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for the legend

    # Save the plot
    plt.savefig(args.output_file, dpi=300)
    print(f"\nPlot saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize WavLM embeddings colored by clinical diagnosis.")
    
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory containing the pooled .npz embedding files."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to the .xlsx file containing clinical metadata."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="diagnosis_umap_visualization.png",
        help="Path to save the output PNG plot."
    )
    parser.add_argument(
        "--color_by_column",
        type=str,
        default="clinical_diagnosis",
        help="The column name in the metadata file to use for coloring the points."
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Which layer's embedding to analyze (0-12). Default is 12."
    )
    
    # You might need to install this: pip install openpyxl
    args = parser.parse_args()
    main(args)


"""
python visualize_by_diagnosis.py \
    --embedding_dir "/home/aleph/redlat_mini/redlat_wavlmbase" \
    --metadata_file "/home/aleph/redlat_mini/total_selected_data_06-02_mini.xlsx" \
    --output_file "diagnosis_clusters_layer12.png" \
    --color_by_column "clinical_diagnosis" \
    --layer 12
"""