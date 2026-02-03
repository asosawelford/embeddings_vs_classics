import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from sklearn.metrics import classification_report, balanced_accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from models import FusionMLP # Make sure this is in your models.py

def main():
    parser = argparse.ArgumentParser(description="Train the fusion model.")
    parser.add_argument('--data_dir', type=str, default='fusion_data')
    parser.add_argument('--classes', nargs='+', default=['AD', 'CN'])
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load pre-processed data
    print("Loading pre-processed fusion data...")
    X_train_acoustic = np.load(os.path.join(args.data_dir, 'train_acoustic.npy'))
    X_train_linguistic = np.load(os.path.join(args.data_dir, 'train_linguistic.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'train_labels.npy'))

    X_val_acoustic = np.load(os.path.join(args.data_dir, 'val_acoustic.npy'))
    X_val_linguistic = np.load(os.path.join(args.data_dir, 'val_linguistic.npy'))
    y_val = np.load(os.path.join(args.data_dir, 'val_labels.npy'))

    X_test_acoustic = np.load(os.path.join(args.data_dir, 'test_acoustic.npy'))
    X_test_linguistic = np.load(os.path.join(args.data_dir, 'test_linguistic.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'test_labels.npy'))

    # 2. Create DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_acoustic, dtype=torch.float32), torch.tensor(X_train_linguistic, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_acoustic, dtype=torch.float32), torch.tensor(X_val_linguistic, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_acoustic, dtype=torch.float32), torch.tensor(X_test_linguistic, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 3. Model, Optimizer, etc.
    model = FusionMLP(
        acoustic_dim=X_train_acoustic.shape[1],
        linguistic_dim=X_train_linguistic.shape[1],
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_val_uar = 0
    patience_counter = 0
    print("\n--- Starting Fusion Model Training ---")
    for epoch in range(args.epochs):
        model.train()
        for acoustic_batch, linguistic_batch, label_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(acoustic_batch.to(device), linguistic_batch.to(device)).squeeze(1)
            loss = criterion(outputs, label_batch.float().to(device))
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for acoustic_batch, linguistic_batch, label_batch in val_loader:
                outputs = model(acoustic_batch.to(device), linguistic_batch.to(device)).squeeze(1)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(label_batch.numpy())
        
        val_uar = balanced_accuracy_score(val_true, val_preds)
        if val_uar > best_val_uar:
            best_val_uar = val_uar
            torch.save(model.state_dict(), "best_fusion_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"  Ep {epoch+1}: Val UAR: {val_uar:.4f} (Best: {best_val_uar:.4f})")
            
        if patience_counter >= args.patience:
            print("Early Stopping.")
            break

    # 5. Final Evaluation
    print("\n--- Evaluating Fusion Model on Test Set ---")
    model.load_state_dict(torch.load("best_fusion_model.pth"))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for acoustic_batch, linguistic_batch, label_batch in test_loader:
            outputs = model(acoustic_batch.to(device), linguistic_batch.to(device)).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(label_batch.numpy())
            
    print("\nFinal Test Results:")
    print(classification_report(test_true, test_preds, target_names=args.classes, digits=4))

if __name__ == '__main__':
    main()

"""
python src/train_fusion.py \
    --data_dir "fusion_data" \
    --lr 1e-4 \
    --hidden_size 4096 \
    --dropout 0.3

"""