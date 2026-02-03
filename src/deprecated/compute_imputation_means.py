import pandas as pd
import json
import argparse

def compute_and_save_means(train_metadata_path, clinical_feature_paths, output_path):
    """
    Calculates the mean for each clinical feature column using only the training data
    and saves the results to a JSON file.

    Args:
        train_metadata_path (str): Path to the training metadata CSV.
        clinical_feature_paths (dict): Dictionary mapping feature names to their CSV paths.
        output_path (str): Path to save the output JSON file.
    """
    print(f"Loading training metadata from: {train_metadata_path}")
    train_df = pd.read_csv(train_metadata_path)
    train_patient_ids = train_df['record_id'].tolist()
    print(f"Found {len(train_patient_ids)} training patients.")

    all_means = {}

    for name, path in clinical_feature_paths.items():
        print(f"Processing feature set: {name} from {path}")
        feature_df = pd.read_csv(path).set_index('id')
        
        # VERY IMPORTANT: Filter the feature dataframe to only include training patients
        train_feature_df = feature_df[feature_df.index.isin(train_patient_ids)]
        
        # Calculate the mean for every column, which automatically ignores NaNs
        feature_means = train_feature_df.mean().to_dict()
        
        # Add these means to our master dictionary
        all_means.update(feature_means)

    print(f"Calculated a total of {len(all_means)} feature means.")
    
    # Save the means to a JSON file
    with open(output_path, 'w') as f:
        json.dump(all_means, f, indent=4)
        
    print(f"✅ Successfully saved imputation means to: {output_path}")

if __name__ == '__main__':
    # This allows you to run the script from the command line
    parser = argparse.ArgumentParser(description="Compute feature means for imputation.")
    parser.add_argument('--train_meta', type=str, required=True, help='Path to train_metadata.csv')
    parser.add_argument('--pitch_csv', type=str, required=True, help='Path to pitch_features.csv')
    parser.add_argument('--timing_csv', type=str, required=True, help='Path to timing_features.csv')
    # Add more arguments if you have more clinical feature files
    parser.add_argument('--output_json', type=str, default='imputation_means.json', help='Path for the output JSON file.')
    
    args = parser.parse_args()

    # Create the dictionary of clinical feature paths
    clinical_paths = {
        'pitch': args.pitch_csv,
        'timing': args.timing_csv
    }

    compute_and_save_means(
        train_metadata_path=args.train_meta,
        clinical_feature_paths=clinical_paths,
        output_path=args.output_json
    )
