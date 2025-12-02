import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import argparse
from tqdm import tqdm
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import our custom modules
# Ensure 'dataloader.py' contains DiskEmbeddingsDataset and pad_collate_fn
from dataloader import DiskEmbeddingsDataset, pad_collate_fn
from models import LearnableStatPoolingMLP

def plot_metrics(history, plot_dir, exp_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{exp_name} Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_uar'], label='Train')
    plt.plot(history['val_uar'], label='Val')
    plt.title(f'{exp_name} UAR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{exp_name}_metrics.png'))
    plt.close()

def main(args):
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    exp_name = f"{args.classes[0]}_vs_{args.classes[1]}_FullSplit"
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print(f"--- Experiment: {exp_name} ---")
    print(f"Device: {device}")

    # 1. Load FULL Metadata & Split
    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    
    # Filter classes
    df = df[df['clinical_diagnosis'].isin(args.classes)]
    
    # Split by Patient ID to avoid leakage
    unique_patients = df[['record_id', 'clinical_diagnosis']].drop_duplicates()
    X = unique_patients['record_id'].values
    y = unique_patients['clinical_diagnosis'].values
    
    # Split: 70% Train, 15% Val, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42 # 0.176 of 0.85 is ~0.15 total
    )
    
    # Create DataFrames
    train_df = df[df['record_id'].isin(X_train)]
    val_df = df[df['record_id'].isin(X_val)]
    test_df = df[df['record_id'].isin(X_test)]
    
    print(f"Split Sizes (Patients): Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")

    # 2. Datasets & Loaders
    train_ds = DiskEmbeddingsDataset(train_df, args.embeddings_dir, args.classes, args.pos_filter)
    val_ds = DiskEmbeddingsDataset(val_df, args.embeddings_dir, args.classes, args.pos_filter)
    test_ds = DiskEmbeddingsDataset(test_df, args.embeddings_dir, args.classes, args.pos_filter)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=pad_collate_fn, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=pad_collate_fn, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=pad_collate_fn, num_workers=4)

    # 3. Model
    model = LearnableStatPoolingMLP(
        num_layers=13, 
        features_per_layer=768, 
        hidden_size=args.hidden_size, 
        dropout_rate=args.dropout
    ).to(device)

    # 4. Optimizer & Scheduler
    base_params = [p for n, p in model.named_parameters() if "layer_averager" not in n]
    weight_params = [p for n, p in model.named_parameters() if "layer_averager" in n]
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr, 'weight_decay': 1e-4}, # Added Regularization
        {'params': weight_params, 'lr': args.lr_weights}
    ])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_uar = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_uar': [], 'val_uar': []}
    save_path = os.path.join(args.results_dir, f"{exp_name}_best.pth")

    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        t_loss = 0
        t_preds, t_true = [], []
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
            if batch is None: continue
            X_b = batch['embedding_feats'].to(device)
            L_b = batch['lengths'].to(device)
            y_b = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            out = model(X_b, L_b).squeeze(1)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            probs = torch.sigmoid(out).detach().cpu().numpy()
            t_preds.extend((probs > 0.5).astype(int))
            t_true.extend(y_b.cpu().numpy())
            
        t_uar = balanced_accuracy_score(t_true, t_preds)
        
        # VAL
        model.eval()
        v_loss = 0
        v_preds, v_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                X_b = batch['embedding_feats'].to(device)
                L_b = batch['lengths'].to(device)
                y_b = batch['label'].float().to(device)
                
                out = model(X_b, L_b).squeeze(1)
                v_loss += criterion(out, y_b).item()
                probs = torch.sigmoid(out).cpu().numpy()
                v_preds.extend((probs > 0.5).astype(int))
                v_true.extend(y_b.cpu().numpy())
        
        v_uar = balanced_accuracy_score(v_true, v_preds)
        
        # Record & Scheduler
        scheduler.step(v_uar)
        
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(val_loader))
        history['train_uar'].append(t_uar)
        history['val_uar'].append(v_uar)
        
        print(f"Ep {epoch+1}: T_UAR {t_uar:.3f} | V_UAR {v_uar:.3f}", end="")
        
        if v_uar > best_val_uar:
            best_val_uar = v_uar
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(" *Best*")
        else:
            patience_counter += 1
            print(f" (Pat {patience_counter}/{args.patience})")
            
        if patience_counter >= args.patience:
            print("Early Stopping.")
            break
            
    plot_metrics(history, args.plots_dir, exp_name)

    # 6. FINAL TEST EVALUATION
    print("\n--- EVALUATING ON HELD-OUT TEST SET ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if batch is None: continue
            X_b = batch['embedding_feats'].to(device)
            L_b = batch['lengths'].to(device)
            y_b = batch['label'].float().to(device)
            
            out = model(X_b, L_b).squeeze(1)
            probs = torch.sigmoid(out).cpu().numpy()
            test_preds.extend((probs > 0.5).astype(int))
            test_true.extend(y_b.cpu().numpy())
            
    # Generate Report
    print("\nFinal Test Results:")
    print(classification_report(test_true, test_preds, target_names=args.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(test_true, test_preds))
    
    with open(os.path.join(args.results_dir, f"{exp_name}_test_report.txt"), "w") as f:
        f.write(classification_report(test_true, test_preds, target_names=args.classes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--embeddings_dir', required=True)
    parser.add_argument('--classes', nargs='+', default=['AD', 'CN'])
    parser.add_argument('--pos_filter', nargs='+', default=None)
    parser.add_argument('--results_dir', default='results_full')
    parser.add_argument('--plots_dir', default='plots_full')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_weights', type=float, default=1e-3)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.6)
    
    args = parser.parse_args()
    main(args)

"""
python src/train_v2.py \
    --metadata "data/metadata_fugu_full_buffer0_emparejado_GLOBAL_additive.csv" \
    --embeddings_dir "/home/aleph/redlat/embeddings/processed_word_embeddings.pt" \

"""
