import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import io
import matplotlib
matplotlib.use('Agg')

# Import our custom modules for this specific task
from dataloader import WordSequenceDataset, pad_collate_fn
from models import LearnableStatPoolingMLP

# Import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

def plot_metrics(epoch_metrics, plot_dir, experiment_name):
    """Plots training and validation loss/UAR over epochs and saves them."""
    if not epoch_metrics:
        print("No metrics to plot.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    
    epochs = [m['epoch'] for m in epoch_metrics]
    train_losses = [m['train_loss'] for m in epoch_metrics]
    val_losses = [m['val_loss'] for m in epoch_metrics]
    train_uars = [m['train_uar'] for m in epoch_metrics]
    val_uars = [m['val_uar'] for m in epoch_metrics]

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title(f'{experiment_name} - Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}_loss.png'))
    plt.close()

    # Plot UAR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_uars, label='Train UAR', marker='o')
    plt.plot(epochs, val_uars, label='Validation UAR', marker='o')
    plt.title(f'{experiment_name} - UAR (Balanced Accuracy) Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('UAR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}_uar.png'))
    plt.close()

    print(f"✅ Plots for Loss and UAR saved to '{plot_dir}'")

def analyze_softmax_weights(model, plot_dir, experiment_name):
    """Analyzes and plots the learned layer weights from LearnableStatPoolingMLP."""
    # The weights are in the 'layer_averager' submodule
    if not hasattr(model, 'layer_averager'):
        print("Model does not have 'layer_averager'. Skipping weight analysis.")
        return ""

    print("\n--- Analyzing Learned Layer Weights (Softmax) ---")
    
    trained_weights = model.layer_averager.weights.data.cpu()
    final_importances = torch.nn.functional.softmax(trained_weights, dim=0)
    
    layer_importances = {i: imp.item() for i, imp in enumerate(final_importances)}
    sorted_layers = sorted(layer_importances.items(), key=lambda item: item[1], reverse=True)

    output_capture = io.StringIO()
    print("\n--- Learned Layer Importance Analysis ---", file=output_capture)
    print("Final importance (softmax weight) for each WavLM layer:", file=output_capture)
    print("Final importance (softmax weight) for each WavLM layer:")
    
    for layer, importance in sorted_layers:
        log_msg = f"  - Layer {layer:<2}: {importance:.4f}  ({importance*100:.2f}%)"
        print(log_msg)
        print(log_msg, file=output_capture)

    # Plot the results
    plt.figure(figsize=(12, 7))
    plt.bar([str(k) for k in layer_importances.keys()], layer_importances.values(), color='darkcyan')
    plt.xlabel("Source Embedding Layer (0-12)")
    plt.ylabel("Learned Importance (Softmax Probability)")
    plt.title(f"Learned Layer Importance - {experiment_name}")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(plot_dir, f'{experiment_name}_layer_weights.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n✅ Layer weight analysis plot saved to '{plot_path}'")
    return output_capture.getvalue()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Experiment Name and Directories
    if args.experiment_name:
            experiment_name = args.experiment_name
    else:
        experiment_name = f"{args.classes[0]}_vs_{args.classes[1]}_LearnablePool"
    print(f"Starting experiment: {experiment_name}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Define the save path for the model based on the experiment name
    save_path = os.path.join(args.results_dir, f'{experiment_name}_best_model.pth')
    print(f"Best model will be saved to: {save_path}")

    # 1. Load the data ONCE into memory
    print(f"Loading main data file into memory from: {args.data_path}")
    raw_word_data = torch.load(args.data_path)
    # Convert list of dicts to a lookup map for efficiency
    embedding_map = {
        (item['id'], item['task']): item['embeddings'] for item in raw_word_data
    }
    print(f"Data loaded. Found embeddings for {len(embedding_map)} unique task-samples.")
    # Free up memory from the list version
    del raw_word_data 
    # --- END CRITICAL CHANGE ---

    # 2. Setup Datasets by passing the IN-MEMORY data
    print("\nSetting up datasets...")
    # Pass the loaded 'embedding_map' to each dataset instance
    train_dataset = WordSequenceDataset(args.train_meta, embedding_map, args.classes)
    val_dataset = WordSequenceDataset(args.val_meta, embedding_map, args.classes)
    
    # Check if datasets were successfully created
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty. Check metadata paths and content.")

    
    # 3. Setup Model
    print("Setting up LearnableStatPoolingMLP model...")
    model = LearnableStatPoolingMLP(
        num_layers=13,
        features_per_layer=768,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate
    ).to(device)
    print(model)

    # 4. Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers, pin_memory=True)

    # 5. Setup Loss and Optimizer with Differential Learning Rate
    criterion = nn.BCEWithLogitsLoss()
    
    print("\nSetting up optimizer with differential learning rates...")
    base_params = [p for name, p in model.named_parameters() if "layer_averager.weights" not in name]
    weight_params = model.layer_averager.weights
    
    optimizer_grouped_parameters = [
        {'params': base_params},
        {'params': weight_params, 'lr': args.lr_weights}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    print(f"  - Base LR: {args.lr}")
    print(f"  - Layer Weights LR: {args.lr_weights}")

    # 6. Training Loop
    best_val_uar = 0.0
    epoch_metrics = []
    patience_counter = 0

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            if batch is None: continue 
            labels = batch['label'].to(device)
            embeddings = batch['embedding_feats'].to(device)
            lengths = batch['lengths'].to(device)

            optimizer.zero_grad()
            outputs = model(embeddings, lengths).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
            train_preds.extend(preds if preds.ndim > 0 else [preds])
            train_labels.extend(labels.cpu().numpy())

        train_uar = balanced_accuracy_score(train_labels, train_preds)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                if batch is None: continue
                labels = batch['label'].to(device)
                embeddings = batch['embedding_feats'].to(device)
                lengths = batch['lengths'].to(device)
                
                outputs = model(embeddings, lengths).squeeze()
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
                val_preds.extend(preds if preds.ndim > 0 else [preds])
                val_labels.extend(labels.cpu().numpy())
        
        val_uar = balanced_accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Train UAR: {train_uar:.4f} | Val Loss: {val_loss/len(val_loader):.4f}, Val UAR: {val_uar:.4f}")

        epoch_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss/len(train_loader), 'train_uar': train_uar, 'val_loss': val_loss/len(val_loader), 'val_uar': val_uar})

        if val_uar > best_val_uar:
            best_val_uar = val_uar
            torch.save(model.state_dict(), save_path)
            print(f"✨ New best model saved with Val UAR: {best_val_uar:.4f} ✨")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\nStopping early. No improvement in Val UAR for {args.patience} epochs.")
                break

    print("\n--- Training Finished ---")
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df.to_csv(os.path.join(args.results_dir, f'{experiment_name}_epoch_metrics.csv'), index=False)
    plot_metrics(epoch_metrics, args.plots_dir, experiment_name)

    # 7. Final Evaluation on Test Set (Patient-Level Aggregation)
    print("\n--- Evaluating on Test Set with Patient-Level Aggregation ---")
    test_dataset = WordSequenceDataset(args.test_meta, embedding_map, args.classes)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=0)
    model.load_state_dict(torch.load(save_path))
    model.eval()

    task_preds_proba, task_true_labels, task_patient_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Set Inference"):
            if batch is None: continue
            embeddings = batch['embedding_feats'].to(device)
            lengths = batch['lengths'].to(device)
            labels = batch['label']
            
            outputs = model(embeddings, lengths).squeeze()
            probabilities = torch.sigmoid(outputs).cpu().numpy()

            task_preds_proba.extend(probabilities if probabilities.ndim > 0 else [probabilities])
            task_true_labels.extend(labels.numpy())
            task_patient_ids.extend(batch['record_id'])

    patient_results_df = pd.DataFrame({'record_id': task_patient_ids, 'predicted_proba': task_preds_proba})
    patient_aggregated_proba = patient_results_df.groupby('record_id')['predicted_proba'].mean()

    test_meta_df = pd.read_csv(args.test_meta)
    patient_labels_map = {args.classes[0]: 0, args.classes[1]: 1}
    patient_true_labels = test_meta_df.drop_duplicates('record_id').set_index('record_id')['clinical_diagnosis'].map(patient_labels_map)
    
    final_df = patient_aggregated_proba.to_frame().join(patient_true_labels).dropna()
    final_df['predicted_label'] = (final_df['predicted_proba'] > 0.5).astype(int)

    patient_true = final_df['clinical_diagnosis'].tolist()
    patient_pred = final_df['predicted_label'].tolist()

    output_capture = io.StringIO()
    print(f"\n--- Patient-Level Test Set Results ({args.classes[0]} vs {args.classes[1]}) ---", file=output_capture)
    print(f"Test UAR (Balanced Accuracy): {balanced_accuracy_score(patient_true, patient_pred):.4f}", file=output_capture)
    print("\nClassification Report:", file=output_capture)
    print(classification_report(patient_true, patient_pred, target_names=args.classes), file=output_capture)
    print("\nConfusion Matrix:", file=output_capture)
    print(confusion_matrix(patient_true, patient_pred), file=output_capture)
    
    final_results_str = output_capture.getvalue()
    print(final_results_str)

    weight_analysis_str = analyze_softmax_weights(model, args.plots_dir, experiment_name)
    with open(os.path.join(args.results_dir, f'{experiment_name}_results.txt'), 'w') as f:
        f.write(final_results_str)
        f.write(weight_analysis_str)
    print(f"✅ Full test results and analysis saved to '{args.results_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a robust classifier on pre-processed word sequence embeddings.")
    
    # --- Paths ---
    parser.add_argument('--data_path', type=str, required=True, help="Path to the 'processed_word_embeddings.pt' file.")
    parser.add_argument('--train_meta', type=str, required=True, help='Path to train_metadata.csv')
    parser.add_argument('--val_meta', type=str, required=True, help='Path to validation_metadata.csv')
    parser.add_argument('--test_meta', type=str, required=True, help='Path to test_metadata.csv') 
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save models, text results, and epoch metrics.') 
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots.') 
    parser.add_argument('--experiment_name', type=str, default=None, help='A specific name for the experiment, used for saving files.')


    # --- Task and Model ---
    parser.add_argument('--classes', nargs='+', default=['AD', 'CN'], help="List of two classes for binary task (e.g., AD CN)")

    # --- Hyperparameters ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate.')
    parser.add_argument('--lr_weights', type=float, default=1e-3, help='A separate, higher learning rate for the layer-weighting parameters.')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden layer size for the final MLP.')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate.')
    parser.add_argument('--patience', type=int, default=20, help='Epochs to wait for improvement before stopping early.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')

    args = parser.parse_args()
    main(args)

"""

python src/train_learnable_pool.py \
    --data_path /home/aleph/redlat/lopera_processed_word_embeddings.pt \
    --train_meta ./data/train_metadata_lopera.csv \
    --val_meta ./data/validation_metadata_lopera.csv \
    --test_meta ./data/test_metadata_lopera.csv \
    --classes AD CN \
    --epochs 100 \
    --batch_size 16 \
    --patience 20 \
    --lr 0.0001 \
    --lr_weights 0.001 \
    --hidden_size 256 \
    --results_dir ./results/AD_CN_LearnablePool \
    --plots_dir ./plots/AD_CN_LearnablePool
"""