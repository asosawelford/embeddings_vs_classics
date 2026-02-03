import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from data_manager import DataManager, AlzheimerDataset
from models import get_model

# --- CONFIGURATION DEFAULTS ---
# You can override these via command line args
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-4

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        # Move data to GPU
        features = batch['features'].to(device)
        labels = batch['label'].to(device).float() # BCE needs float labels

        optimizer.zero_grad()
        
        # Forward
        logits = model(features).squeeze(1) # (Batch, 1) -> (Batch,)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        
        # Store for metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, np.round(all_preds))
    except:
        auc = 0.5 # Handle edge case of single-class batch
        acc = 0.0

    return avg_loss, acc, auc

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device).float()

            logits = model(features).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, np.round(all_preds))
    except:
        auc = 0.5
        acc = 0.0

    return avg_loss, acc, auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Single Split Training on {device}")
    print(f"   Model: {args.model_type}")
    print(f"   Tasks: {args.tasks}")

    # 1. Initialize Data Manager
    manager = DataManager(args.metadata, target_groups=['CN', 'AD'])
    
    # 2. Split Patients (80/20)
    all_pats = manager.unique_patients
    n_train = int(len(all_pats) * 0.8)
    train_pats = all_pats[:n_train]
    test_pats = all_pats[n_train:]
    
    train_df, test_df = manager.split_patients(train_pats, test_pats)
    print(f"   Split: {len(train_pats)} Train Patients / {len(test_pats)} Test Patients")

    # 3. Load Data based on Type
    input_dim = None # Will be determined dynamically
    
    if args.model_type in ['wavlm', 'roberta']:
        # Load Embeddings
        print(f"   Loading embeddings from: {args.embedding_dir}")
        X_tr, y_tr, ids_tr, tasks_tr = manager.load_embeddings(train_df, args.embedding_dir, args.tasks, args.model_type)
        X_te, y_te, ids_te, tasks_te = manager.load_embeddings(test_df, args.embedding_dir, args.tasks, args.model_type)
        
        # Determine Input Dim for RoBERTa (WavLM is handled by wrapper)
        if args.model_type == 'roberta':
            input_dim = X_tr.shape[1] # Should be 768

    elif args.model_type == 'classic':
        # Load CSV features
        print(f"   Loading classic features from: {args.classic_csv}")
        if args.classic_subset:
            print(f"   Subset: {args.classic_subset}")

        (X_tr, y_tr, ids_tr, tasks_tr), (X_te, y_te, ids_te, tasks_te) = manager.load_classic_features(
            train_df, test_df, args.classic_csv, args.tasks, k=5, subset=args.classic_subset
        )
        input_dim = X_tr.shape[1]
        print(f"   Classic Input Dimension: {input_dim}")

    # 4. Create Datasets & Loaders
    train_ds = AlzheimerDataset(X_tr, y_tr, ids_tr, tasks_tr)
    test_ds = AlzheimerDataset(X_te, y_te, ids_te, tasks_te)

    train_loader = DataLoader(train_ds, batch_size=DEFAULT_BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=DEFAULT_BATCH, shuffle=False)
    
    print(f"   Train Samples: {len(train_ds)} | Test Samples: {len(test_ds)}")

    # 5. Initialize Model
    model = get_model(args.model_type, input_dim=input_dim, dropout=0.3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    print("\n--- Training Start ---")
    best_auc = 0.0
    
    for epoch in range(DEFAULT_EPOCHS):
        tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, test_loader, criterion, device)
        
        # Simple logging
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f} AUC: {tr_auc:.2f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f} AUC: {val_auc:.2f}")

        if val_auc > best_auc:
            best_auc = val_auc
            # Optional: torch.save(model.state_dict(), 'best_model.pth')

    print(f"\n✅ Training Complete. Best Test AUC: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=['wavlm', 'roberta', 'classic'])
    parser.add_argument("--metadata", type=str, required=True)
    
    # Optional arguments depending on model_type
    parser.add_argument("--embedding_dir", type=str, default=None)
    parser.add_argument("--classic_csv", type=str, default=None)
    
    # Tasks (can pass multiple)
    parser.add_argument("--tasks", nargs='+', default=['CraftIm', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu'])
    parser.add_argument("--classic_subset", type=str, choices=['audio', 'language'], default=None, 
                        help="Filter classic features by type (audio or language)")
    
    args = parser.parse_args()
    
    # Validation of args
    if args.model_type in ['wavlm', 'roberta'] and not args.embedding_dir:
        parser.error("--embedding_dir is required for wavlm/roberta")
    if args.model_type == 'classic' and not args.classic_csv:
        parser.error("--classic_csv is required for classic models")
        
    main(args)