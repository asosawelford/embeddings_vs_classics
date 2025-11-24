import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# Assuming your models and dataloader are in the src/ directory
from src.dataloader import PatientTaskDataset, collate_fn
from src.models import WeightedEmbeddingMLP # Make sure this is the correct model class

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# --- Paths to Your Trained Specialist Models ---
# !!! IMPORTANT: Update these paths to point to your actual trained .pth files !!!
MODEL_PATHS = {
    "video_retelling": "models/specialists/fluency_model.pth",
    "fluency": "models/specialists/fluency_model.pth",
    "short_story": "models/specialists/short_story_model.pth"
}

# --- Task Definitions ---
# This must match the tasks each model was trained on.
TASK_GROUPS = {
    "video_retelling": ['Fugu'],
    "fluency": ['Phonological', 'Phonological2', 'Semantic', 'Semantic2'],
    "short_story": ['CraftIm']
}

# --- Test Data and Model Parameters ---
# Use one of your existing test sets for evaluation
TEST_META_PATH = 'data/final_split/final_test_set.csv'
BATCH_SIZE = 32 # <-- FIX #1: Moved batch_size to its own constant
# These params must match what the models were trained with
MODEL_PARAMS = {
    "features_per_layer": 768,
    "hidden_size": 1024, # Assuming this was the final size
    "num_layers": 13,
    "dropout_rate": 0.4 # Assuming this was the final dropout
}
# Other dataloader params
DATASET_PARAMS = {
    "embedding_path": "/home/aleph/redlat/walm-time-pooled",
    "imputation_means_path": "data/test_metadata_lopera.csv",
    "scaling_params_path": "data/test_metadata_lopera.csv",
    "classes_to_load": ["CN", "AD"],
    # mode is passed separately inside the loop
}

# ==============================================================================
# --- 2. ENSEMBLE EVALUATION SCRIPT ---
# ==============================================================================

def get_specialist_predictions(model, dataloader, device):
    """Run inference for one specialist model and return predictions per patient."""
    model.eval()
    patient_predictions = {}
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            
            record_ids = batch['record_id']
            features = batch['embedding_feats'].to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

            if probabilities.ndim == 0:
                probabilities = [probabilities.item()]

            for i, record_id in enumerate(record_ids):
                if record_id not in patient_predictions:
                    patient_predictions[record_id] = []
                patient_predictions[record_id].append(probabilities[i])

    # Aggregate task-level predictions to patient-level by averaging
    for record_id, preds in patient_predictions.items():
        patient_predictions[record_id] = np.mean(preds)
        
    return patient_predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Ensemble Evaluation on device: {device} ---")

    print("Loading specialist models...")
    models = {}
    for group_name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"!!! FATAL: Model for '{group_name}' not found at {path}. Exiting.")
            return
        model = WeightedEmbeddingMLP(**MODEL_PARAMS)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        models[group_name] = model
    
    print("Creating specialist dataloaders for the test set...")
    dataloaders = {}
    for group_name, tasks in TASK_GROUPS.items():
        # <-- FIX #2: Pass the correct dictionary to the Dataset
        dataset = PatientTaskDataset(
            metadata_path=TEST_META_PATH,
            clinical_feature_paths={}, # Not needed for embedding mode
            mode='embedding', # Pass mode directly
            tasks_to_load=tasks,
            **DATASET_PARAMS 
        )
        # <-- FIX #3: Use the BATCH_SIZE constant here
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        dataloaders[group_name] = loader

    print("Running inference with each specialist model...")
    specialist_predictions = {}
    for group_name, model in models.items():
        print(f"  Getting predictions for: {group_name}")
        loader = dataloaders[group_name]
        specialist_predictions[group_name] = get_specialist_predictions(model, loader, device)

    print("\n--- Aggregating predictions and evaluating ensemble ---")
    
    test_meta_df = pd.read_csv(TEST_META_PATH)
    patient_labels_map = {"CN": 0, "AD": 1}
    patient_true_labels_df = test_meta_df[['record_id', 'clinical_diagnosis']].drop_duplicates().set_index('record_id')
    patient_true_labels_df['true_label'] = patient_true_labels_df['clinical_diagnosis'].map(patient_labels_map)

    all_preds_df = pd.DataFrame(specialist_predictions).join(patient_true_labels_df['true_label'], how='inner')

    print("\n--- [RESULTS] Soft Voting (Averaging Probabilities) ---")
    all_preds_df['soft_vote_proba'] = all_preds_df[list(TASK_GROUPS.keys())].mean(axis=1)
    all_preds_df['soft_vote_label'] = (all_preds_df['soft_vote_proba'] > 0.5).astype(int)
    soft_uar = balanced_accuracy_score(all_preds_df['true_label'], all_preds_df['soft_vote_label'])
    print(f"Test UAR: {soft_uar:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_preds_df['true_label'], all_preds_df['soft_vote_label'], target_names=["CN", "AD"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_preds_df['true_label'], all_preds_df['soft_vote_label']))

    print("\n--- [RESULTS] Hard Voting (Majority Vote) ---")
    pred_labels_df = (all_preds_df[list(TASK_GROUPS.keys())] > 0.5).astype(int)
    all_preds_df['hard_vote_label'] = pred_labels_df.mode(axis=1)[0]
    hard_uar = balanced_accuracy_score(all_preds_df['true_label'], all_preds_df['hard_vote_label'])
    print(f"Test UAR: {hard_uar:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_preds_df['true_label'], all_preds_df['hard_vote_label'], target_names=["CN", "AD"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_preds_df['true_label'], all_preds_df['hard_vote_label']))


if __name__ == "__main__":
    main()
