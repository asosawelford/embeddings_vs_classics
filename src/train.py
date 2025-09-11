import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import functools
import warnings

# Import our custom modules
from dataloader import PatientTaskDataset, collate_fn
from models import ExplainableMLP
# We will add EmbeddingMLP and FusionANN here later
# from models import EmbeddingMLP, FusionANN

# Import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

def get_clinical_input_size(dataset):
    # ... (this function remains the same) ...
    max_len = 0
    # Create a temporary dataset instance without trying to load all data initially
    # to avoid unnecessary memory usage and only get a single sample.
    # This might require a temporary change to PatientTaskDataset if it's too aggressive.
    # For now, let's assume it works.
    
    # A safer way to get the max length without loading everything for real:
    # If you know your features are consistent across files, you can just
    # load one patient's data for one task and get its length.
    
    # Alternatively, the scaling_params.json already contains 'feature_dim'
    # which IS the max length determined from the training set.
    # So, we can directly read that.
    
    # For now, we'll keep iterating, as it's already implemented and working.
    # If performance becomes an issue, we can optimize by reading from scaling_params.json
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample and 'clinical_feats' in sample:
            current_len = len(sample['clinical_feats'])
            if current_len > max_len:
                max_len = current_len
    return max_len


def main(args):
    # Silence pandas future warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="dataloader")

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Datasets
    print("Setting up datasets...")
    # These datasets are used for model training and validation
    train_dataset = PatientTaskDataset(
        metadata_path=args.train_meta,
        clinical_feature_paths=args.clinical_paths,
        embedding_path=args.embedding_path,
        mode=args.mode,
        imputation_means_path=args.imputation_means,
        scaling_params_path=args.scaling_params,
        classes_to_load=args.classes,
        embedding_layers=args.embedding_layers
    )

    val_dataset = PatientTaskDataset(
        metadata_path=args.val_meta,
        clinical_feature_paths=args.clinical_paths,
        embedding_path=args.embedding_path,
        mode=args.mode,
        imputation_means_path=args.imputation_means,
        scaling_params_path=args.scaling_params,
        classes_to_load=args.classes,
        embedding_layers=args.embedding_layers
    )

    # 3. Setup Model
    print(f"Setting up model for mode: {args.mode}")
    collate_function = None # Initialize
    input_size = None 
    
    if args.mode == 'clinical':
        input_size = get_clinical_input_size(train_dataset) # Get max length from training data
        print(f"Determined max clinical feature length (model input size): {input_size}")
        
        model = ExplainableMLP(
            input_size=input_size, 
            hidden_size=args.hidden_size, 
            dropout_rate=args.dropout
        ).to(device)

        collate_function = functools.partial(collate_fn, fixed_clinical_len=input_size)
    
    # Placeholder for future models
    # elif args.mode == 'embedding':
    #     embedding_feature_dim = len(args.embedding_layers) * 768
    #     model = EmbeddingMLP(input_size=embedding_feature_dim, ...).to(device)
    #     collate_function = functools.partial(collate_fn, fixed_clinical_len=None) # No clinical padding needed
    # elif args.mode == 'fusion':
    #     # Need both clinical_input_size and embedding_feature_dim
    #     clinical_input_size_for_fusion = get_clinical_input_size(train_dataset)
    #     embedding_feature_dim_for_fusion = len(args.embedding_layers) * 768
    #     model = FusionANN(clinical_input_size=clinical_input_size_for_fusion, 
    #                       embedding_input_size=embedding_feature_dim_for_fusion, ...).to(device)
    #     collate_function = functools.partial(collate_fn, fixed_clinical_len=clinical_input_size_for_fusion)

    else:
        raise ValueError(f"Mode '{args.mode}' not recognized.")
        
    print(model)

    # 4. Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_function)

    # 5. Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 6. Training Loop
    best_val_uar = 0.0
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            if batch is None: continue # Skip if batch is empty
            
            features = batch['clinical_feats'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            preds = torch.sigmoid(outputs).round().squeeze().detach().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_uar = balanced_accuracy_score(train_labels, train_preds)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                if batch is None: continue

                features = batch['clinical_feats'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                val_loss += loss.item()

                preds = torch.sigmoid(outputs).round().squeeze().detach().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_uar = balanced_accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train UAR: {train_uar:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val UAR: {val_uar:.4f}")

        if val_uar > best_val_uar:
            best_val_uar = val_uar
            torch.save(model.state_dict(), args.save_path)
            print(f"✨ New best model saved with Val UAR: {best_val_uar:.4f} ✨")

    print("\nTraining finished.")

    # --- 7. Final Evaluation on Test Set (Patient-Level Aggregation) ---
    print("\n--- Evaluating on Test Set ---")
    
    # 7.1. Setup Test Dataset and DataLoader
    test_dataset = PatientTaskDataset(
        metadata_path=args.test_meta, # <--- NEW: Using test_meta
        clinical_feature_paths=args.clinical_paths,
        embedding_path=args.embedding_path,
        mode=args.mode,
        imputation_means_path=args.imputation_means,
        scaling_params_path=args.scaling_params,
        classes_to_load=args.classes,
        embedding_layers=args.embedding_layers
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_function)

    # 7.2. Load Best Model
    model.load_state_dict(torch.load(args.save_path))
    model.eval() # Set to evaluation mode

    # 7.3. Collect Predictions per Task-Sample
    task_preds_proba = [] # Store probabilities
    task_true_labels = [] # Store true labels for each task-sample
    task_patient_ids = [] # Store patient IDs for each task-sample

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test Set Inference"):
            if batch is None: continue

            features = batch['clinical_feats'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            # Apply sigmoid to get probabilities for binary classification
            probabilities = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
            
            task_preds_proba.extend(probabilities)
            task_true_labels.extend(labels.cpu().numpy())
            
            # Get corresponding record_ids for this batch
            # We need to map batch index back to sample index in test_dataset
            batch_patient_ids = [test_dataset.samples[idx][0] for idx in range(i * args.batch_size, min((i + 1) * args.batch_size, len(test_dataset)))]
            task_patient_ids.extend(batch_patient_ids)

    # 7.4. Aggregate Predictions at Patient Level
    print("\nAggregating predictions at patient level...")
    patient_results_df = pd.DataFrame({
        'record_id': task_patient_ids,
        'predicted_proba': task_preds_proba,
        'true_label_task': task_true_labels # Keep this for consistency, but not used for patient-level truth
    })

    # Get true patient-level diagnoses from the *original* metadata (or test_metadata specifically)
    # This ensures we get one true diagnosis per patient, regardless of tasks.
    # Load the test metadata (which should contain unique record_id and clinical_diagnosis)
    test_meta_df = pd.read_csv(args.test_meta)
    
    # Filter and map labels for the specific binary task, as done in PatientTaskDataset
    test_meta_df = test_meta_df[test_meta_df['clinical_diagnosis'].isin(args.classes)]
    patient_labels_map = {args.classes[0]: 0, args.classes[1]: 1}
    test_meta_df['true_label_patient'] = test_meta_df['clinical_diagnosis'].map(patient_labels_map)
    
    # Ensure one true label per patient_id (drop duplicates if needed, based on your metadata structure)
    patient_true_labels = test_meta_df[['record_id', 'true_label_patient']].drop_duplicates().set_index('record_id')

    # Group by patient and average probabilities
    patient_aggregated_proba = patient_results_df.groupby('record_id')['predicted_proba'].mean()

    # Join with true patient labels
    final_patient_evaluation_df = patient_aggregated_proba.to_frame().join(patient_true_labels)
    final_patient_evaluation_df.reset_index(inplace=True)

    # Make final binary prediction (e.g., threshold at 0.5)
    final_patient_evaluation_df['predicted_label_patient'] = (final_patient_evaluation_df['predicted_proba'] > 0.5).astype(int)

    # 7.5. Calculate Patient-Level Metrics
    patient_true = final_patient_evaluation_df['true_label_patient'].tolist()
    patient_pred = final_patient_evaluation_df['predicted_label_patient'].tolist()

    test_acc = accuracy_score(patient_true, patient_pred)
    test_uar = balanced_accuracy_score(patient_true, patient_pred)
    
    # Get original class names for classification report
    target_names = args.classes
    
    print(f"\n--- Patient-Level Test Set Results ({target_names[0]} vs {target_names[1]}) ---")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test UAR (Balanced Accuracy): {test_uar:.4f}")
    print("\nClassification Report:")
    print(classification_report(patient_true, patient_pred, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(patient_true, patient_pred))

# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classifier for AD/FTD/CN discrimination.")
    
    # --- Paths ---
    parser.add_argument('--train_meta', type=str, required=True, help='Path to train_metadata.csv')
    parser.add_argument('--val_meta', type=str, required=True, help='Path to validation_metadata.csv')
    parser.add_argument('--test_meta', type=str, required=True, help='Path to test_metadata.csv') # <--- NEW ARGUMENT
    parser.add_argument('--pitch_csv', type=str, required=True, help='Path to pitch_features.csv')
    parser.add_argument('--timing_csv', type=str, required=True, help='Path to timing_features.csv')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to embedding folder')
    parser.add_argument('--imputation_means', type=str, required=True, help='Path to imputation_means.json')
    parser.add_argument('--scaling_params', type=str, required=True, help='Path to scaling_params.json')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save the best model')

    # --- Task and Model ---
    parser.add_argument('--classes', nargs='+', required=True, help="List of two classes for binary task (e.g., CN AD)")
    parser.add_argument('--mode', type=str, default='clinical', choices=['clinical', 'embedding', 'fusion'], help='Model mode')
    parser.add_argument('--embedding_layers', type=int, nargs='+', default=[0, 8, 11], help='WavLM layers to use')
    
    # --- Hyperparameters ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size for MLP')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Convert paths to a dictionary for the dataloader
    args.clinical_paths = {
        'pitch': args.pitch_csv,
        'timing': args.timing_csv
    }
    
    main(args)