import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from tqdm import tqdm
import functools
import warnings
import os
import json
import matplotlib.pyplot as plt
import io

# Import our custom modules
from dataloader import PatientTaskDataset, collate_fn
from models import ExplainableMLP, FusionANN, WeightedEmbeddingMLP

# Import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

# Optimized get_clinical_input_size to read from scaling_params.json
def get_clinical_input_size(scaling_params_path):
    with open(scaling_params_path, 'r') as f:
        scaling_data = json.load(f)
    return scaling_data['feature_dim']


def plot_metrics(epoch_metrics, plot_dir, experiment_name):
    """Plots training and validation loss/UAR over epochs."""
    epochs = [m['epoch'] for m in epoch_metrics]
    train_losses = [m['train_loss'] for m in epoch_metrics]
    val_losses = [m['val_loss'] for m in epoch_metrics]
    train_uars = [m['train_uar'] for m in epoch_metrics]
    val_uars = [m['val_uar'] for m in epoch_metrics]

    os.makedirs(plot_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'{experiment_name} - Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}_loss.png'))
    plt.close()

    # Plot UAR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_uars, label='Train UAR')
    plt.plot(epochs, val_uars, label='Validation UAR')
    plt.title(f'{experiment_name} - UAR Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('UAR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}_uar.png'))
    plt.close()

    print(f"✅ Plots saved to {plot_dir}")

def analyze_softmax_weights(model, embedding_layers_used, plot_dir, experiment_name):
    print("\n--- Analyzing Learned Layer Weights (Softmax) ---")
    
    # Get the raw learned weights from the model
    trained_weights = model.weighted_average.weights.data.cpu()
    
    # Apply softmax to get the final probabilities
    final_importances = torch.nn.functional.softmax(trained_weights, dim=0)
    
    layer_importances = {layer_idx: imp.item() for layer_idx, imp in zip(embedding_layers_used, final_importances)}
    
    # Sort for printing
    sorted_layers = sorted(layer_importances.items(), key=lambda item: item[1], reverse=True)

    print("Final learned importance (softmax weight) for each embedding layer:")
    for layer, importance in sorted_layers:
        print(f"  - Layer {layer:<2}: {importance:.4f}  ({importance*100:.2f}%)")

    # Plot the results
    plt.figure(figsize=(12, 7))
    plt.bar([str(k) for k in layer_importances.keys()], layer_importances.values(), color='mediumseagreen')
    plt.xlabel("Source Embedding Layer (from WavLM)")
    plt.ylabel("Learned Importance (Softmax Probability)")
    plt.title("Learned Importance of Source Embedding Layers")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(plot_dir, f'{experiment_name}_layer_weights.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n✅ Layer weight analysis plot saved to {plot_path}")


def main(args):
    # Silence pandas future warning
    warnings.filterwarnings("ignore", category=FutureWarning, module="dataloader")

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine experiment name for saving files
    experiment_name = f"{args.classes[0]}_vs_{args.classes[1]}_{args.mode}"
    # Append embedding layers if mode is embedding or fusion, for specific save path
    if args.mode in ['embedding', 'fusion']:
        experiment_name += f"_layers_{'_'.join(map(str, args.embedding_layers))}"

    print(f"Starting experiment: {experiment_name}")

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # 2. Setup Datasets
    print("Setting up datasets...")
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
    
    if args.mode == 'clinical':
        input_size = get_clinical_input_size(args.scaling_params)
        print(f"Clinical feature input size: {input_size}")
        
        model = ExplainableMLP(
            input_size=input_size, 
            hidden_size=args.hidden_size, 
            dropout_rate=args.dropout
        ).to(device)

        collate_function = functools.partial(collate_fn, fixed_clinical_len=input_size)
    
    elif args.mode == 'embedding': # <--- NEW MODE HANDLING
        embedding_input_size = len(args.embedding_layers) * 768
        features_per_layer = 768 # For WavLM-base
        num_layers = len(args.embedding_layers)
        print(f"Using WeightedEmbeddingMLP with {num_layers} layers.")

        
        model = WeightedEmbeddingMLP(
            features_per_layer=features_per_layer,
            hidden_size=args.hidden_size,
            num_layers=num_layers,
            dropout_rate=args.dropout
        ).to(device)

        collate_function = functools.partial(collate_fn, fixed_clinical_len=None) # No clinical features, so no fixed_clinical_len
    
    elif args.mode == 'fusion': # <--- NEW FUSION MODE HANDLING
        clinical_input_size = get_clinical_input_size(args.scaling_params)
        embedding_input_size = len(args.embedding_layers) * 768
        print(f"Fusion model input sizes: Clinical={clinical_input_size}, Embedding={embedding_input_size}")
        
        model = FusionANN(
            clinical_input_size=clinical_input_size,
            embedding_input_size=embedding_input_size,
            # You can add more argparse args for these hyperparameters
            linear_hidden_size=args.hidden_size,
            dropout_rate=args.dropout
        ).to(device)

        collate_function = functools.partial(collate_fn, fixed_clinical_len=clinical_input_size)

    else:
        raise ValueError(f"Mode '{args.mode}' not recognized or not implemented yet.")
        
    print(model)

    # 4. Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function,
                              num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_function,
                              num_workers=8, pin_memory=True)

    # 5. Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    # 5. Setup Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()

    # --- Implement Differential Learning Rate for the WeightedAverage layer ---
    print("\nSetting up optimizer with differential learning rates for layer weights...")
    
    # Separate the parameters into two groups
    # Group 1: All parameters EXCEPT the special weights
    base_params = [p for name, p in model.named_parameters() if "weighted_average.weights" not in name]
    
    # Group 2: ONLY the special weights from the WeightedAverage layer
    weight_params = model.weighted_average.weights
    
    # Create the parameter groups for the optimizer
    optimizer_grouped_parameters = [
        {'params': base_params}, # Gets the default LR from args.lr
        {'params': weight_params, 'lr': args.lr_weights} # Gets the special, higher LR
    ]

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    
    print(f"  - Base LR: {args.lr}")
    print(f"  - WeightedAverage LR: {args.lr_weights}")

    # 6. Training Loop
    best_val_uar = 0.0
    epoch_metrics = []

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            if batch is None: continue 
            labels = batch['label'].to(device)
            
            # --- FEATURE EXTRACTION FROM BATCH (UPDATED FOR FUSION) ---
            if args.mode == 'fusion':
                features_clinical = batch['clinical_feats'].to(device)
                features_embedding = batch['embedding_feats'].to(device)
                outputs = model(features_clinical, features_embedding)
            else: # For clinical or embedding mode
                features = batch.get('clinical_feats', batch.get('embedding_feats')).to(device)
                outputs = model(features)

            optimizer.zero_grad()
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
            
                labels = batch['label'].to(device)
                
                if args.mode == 'fusion':
                    features_clinical = batch['clinical_feats'].to(device)
                    features_embedding = batch['embedding_feats'].to(device)
                    outputs = model(features_clinical, features_embedding)
                else:
                    features = batch.get('clinical_feats', batch.get('embedding_feats')).to(device)
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

        # Store epoch metrics
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_uar': train_uar,
            'val_loss': val_loss / len(val_loader),
            'val_uar': val_uar
        })

        if val_uar > best_val_uar:
            best_val_uar = val_uar
            torch.save(model.state_dict(), args.save_path)
            print(f"✨ New best model saved with Val UAR: {best_val_uar:.4f} ✨")

    print("\nTraining finished.")
    
    # Save epoch metrics to CSV
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_csv_path = os.path.join(args.results_dir, f'{experiment_name}_epoch_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✅ Epoch metrics saved to {metrics_csv_path}")

    # Plot metrics
    plot_metrics(epoch_metrics, args.plots_dir, experiment_name)


    # --- 7. Final Evaluation on Test Set (Patient-Level Aggregation) ---
    print("\n--- Evaluating on Test Set ---")
    
    test_dataset = PatientTaskDataset(
        metadata_path=args.test_meta, 
        clinical_feature_paths=args.clinical_paths,
        embedding_path=args.embedding_path,
        mode=args.mode,
        imputation_means_path=args.imputation_means,
        scaling_params_path=args.scaling_params,
        classes_to_load=args.classes,
        embedding_layers=args.embedding_layers
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_function)

    # Load Best Model
    model.load_state_dict(torch.load(args.save_path))
    model.eval() # Set to evaluation mode

    # Collect Predictions per Task-Sample
    task_preds_proba = [] 
    task_true_labels = [] 
    task_patient_ids = [] 

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test Set Inference"):
            if batch is None: continue
            batch_patient_ids = batch['record_id']
            if batch is None: continue
            
            labels = batch['label'].to(device)
            
            if args.mode == 'fusion':
                features_clinical = batch['clinical_feats'].to(device)
                features_embedding = batch['embedding_feats'].to(device)
                outputs = model(features_clinical, features_embedding)
            else:
                features = batch.get('clinical_feats', batch.get('embedding_feats')).to(device)
                outputs = model(features)
            
            probabilities = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()

            # If batch size is 1, probabilities might not be an iterable array
            if probabilities.ndim == 0:
                probabilities = [probabilities.item()]

            task_preds_proba.extend(probabilities)
            task_true_labels.extend(labels.cpu().numpy())
            task_patient_ids.extend(batch_patient_ids) # Extend with IDs from batch

    # 7.4. Aggregate Predictions at Patient Level
    print("\nAggregating predictions at patient level...")
    patient_results_df = pd.DataFrame({
        'record_id': task_patient_ids,
        'predicted_proba': task_preds_proba,
        'true_label_task': task_true_labels 
    })

    test_meta_df = pd.read_csv(args.test_meta)
    test_meta_df = test_meta_df[test_meta_df['clinical_diagnosis'].isin(args.classes)]
    patient_labels_map = {args.classes[0]: 0, args.classes[1]: 1}
    test_meta_df['true_label_patient'] = test_meta_df['clinical_diagnosis'].map(patient_labels_map)
    patient_true_labels = test_meta_df[['record_id', 'true_label_patient']].drop_duplicates().set_index('record_id')

    patient_aggregated_proba = patient_results_df.groupby('record_id')['predicted_proba'].mean()

    final_patient_evaluation_df = patient_aggregated_proba.to_frame().join(patient_true_labels)
    final_patient_evaluation_df.reset_index(inplace=True)

    final_patient_evaluation_df['predicted_label_patient'] = (final_patient_evaluation_df['predicted_proba'] > 0.5).astype(int)

    # 7.5. Calculate Patient-Level Metrics
    patient_true = final_patient_evaluation_df['true_label_patient'].tolist()
    patient_pred = final_patient_evaluation_df['predicted_label_patient'].tolist()

    test_acc = accuracy_score(patient_true, patient_pred)
    test_uar = balanced_accuracy_score(patient_true, patient_pred)
    target_names = args.classes
    
    # --- Capture and Save Results ---
    # Use io.StringIO to capture print output
    output_capture = io.StringIO()
    
    print(f"\n--- Patient-Level Test Set Results ({target_names[0]} vs {target_names[1]}) ---", file=output_capture)
    print(f"Test Accuracy: {test_acc:.4f}", file=output_capture)
    print(f"Test UAR (Balanced Accuracy): {test_uar:.4f}", file=output_capture)
    print("\nClassification Report:", file=output_capture)
    print(classification_report(patient_true, patient_pred, target_names=target_names), file=output_capture)
    print("\nConfusion Matrix:", file=output_capture)
    print(confusion_matrix(patient_true, patient_pred), file=output_capture)

    # Print to console and save to file
    final_results_str = output_capture.getvalue()
    print(final_results_str) # Also print to console

    results_file_path = os.path.join(args.results_dir, f'{experiment_name}_results.txt')
    with open(results_file_path, 'w') as f:
        f.write(final_results_str)
    print(f"✅ Test results saved to {results_file_path}")

     # --- THE WEIGHT ANALYSIS CALL ---
    if args.mode == 'embedding':
        # Use our new analysis function
        analyze_softmax_weights(
            model=model,
            embedding_layers_used=args.embedding_layers,
            plot_dir=args.plots_dir,
            experiment_name=experiment_name
        )

# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a classifier for AD/FTD/CN discrimination.")
    
    # --- Paths ---
    parser.add_argument('--train_meta', type=str, required=True, help='Path to train_metadata.csv')
    parser.add_argument('--val_meta', type=str, required=True, help='Path to validation_metadata.csv')
    parser.add_argument('--test_meta', type=str, required=True, help='Path to test_metadata.csv') 
    parser.add_argument('--pitch_csv', type=str, required=True, help='Path to pitch_features.csv')
    parser.add_argument('--timing_csv', type=str, required=True, help='Path to timing_features.csv')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to embedding folder')
    parser.add_argument('--imputation_means', type=str, required=True, help='Path to imputation_means.json')
    parser.add_argument('--scaling_params', type=str, required=True, help='Path to scaling_params.json')
    parser.add_argument('--save_path', type=str, default='_dynamic_path', help='Path to save the best model (will be constructed based on experiment name and results_dir)')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save text results and epoch metrics') 
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots') 
    parser.add_argument('--lr_weights', type=float, default=1e-3, help='A separate, higher learning rate for the WeightedAverage layer.')

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