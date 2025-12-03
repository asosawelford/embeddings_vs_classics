import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ================= CONFIGURATION =================
DEFAULT_TASKS = ['CraftIm', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
DEFAULT_FEATURES = ["concreteness", "granularity", "mean_num_syll", "mean_phon_neigh", "mean_log_frq", "pitch", "timing"]
# =================================================

class FlexibleMLP(nn.Module):
    """ An MLP with a variable number of hidden layers. """
    def __init__(self, input_dim, hidden_size, num_hidden_layers, dropout_rate):
        super().__init__()
        
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden Layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output Layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        self.input_layer = self.network[0] # For feature importance

    def forward(self, x):
        return self.network(x)

def analyze_feature_importance(model, feature_names, n_top=20):
    # This function remains unchanged.
    print("\n" + "="*40 + "\n=== FEATURE IMPORTANCE ANALYSIS ===\n" + "="*40)
    weights_matrix = model.input_layer.weight.data.cpu().numpy()
    importance_scores = np.sum(np.abs(weights_matrix), axis=0)
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance_scores}).sort_values('importance', ascending=False)
    print(f"Top {n_top} most important features:\n")
    print(importance_df.head(n_top).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Sweep hyperparameters for the linguistic classifier.")
    # Data Paths
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--features_csv', type=str, required=True)
    # Feature Selection
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--features', nargs='+', default=DEFAULT_FEATURES)
    parser.add_argument('--classes', nargs='+', default=['AD', 'CN'])
    # Output
    parser.add_argument('--results_dir', type=str, default='sweep_results')
    # --- SWEEPABLE HYPERPARAMETERS ---
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # Training Control
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Create a unique name for this run ---
    exp_name = f"layers_{args.num_hidden_layers}_size_{args.hidden_size}_lr_{args.lr}_do_{args.dropout}"
    run_results_dir = os.path.join(args.results_dir, exp_name)
    os.makedirs(run_results_dir, exist_ok=True)
    print(f"\n--- Starting Run: {exp_name} ---")

    # 1. Load, Merge, and Split Data (Same as before)
    meta_df = pd.read_csv(args.metadata, low_memory=False)
    features_df = pd.read_csv(args.features_csv, low_memory=False)
    df = pd.merge(meta_df[['record_id', 'clinical_diagnosis']], features_df, on='record_id', how='inner')
    feature_cols = [col for col in df.columns if any(task in col for task in args.tasks) and any(feat in col for feat in args.features)]
    
    df = df.dropna(subset=['clinical_diagnosis'])
    df = df[df['clinical_diagnosis'].isin(args.classes)]
    unique_patients = df[['record_id', 'clinical_diagnosis']].drop_duplicates()
    
    train_val_ids, test_ids, y_train_val_labels, _ = train_test_split(unique_patients['record_id'], unique_patients['clinical_diagnosis'], test_size=0.15, stratify=unique_patients['clinical_diagnosis'], random_state=42)
    train_ids, val_ids, _, _ = train_test_split(train_val_ids, y_train_val_labels, test_size=0.1765, stratify=y_train_val_labels, random_state=42)

    train_df, val_df, test_df = df[df.record_id.isin(train_ids)].copy(), df[df.record_id.isin(val_ids)].copy(), df[df.record_id.isin(test_ids)].copy()

    le = LabelEncoder()
    le.fit(df['clinical_diagnosis'])
    train_df['label'] = le.transform(train_df['clinical_diagnosis'])
    val_df['label'] = le.transform(val_df['clinical_diagnosis'])
    test_df['label'] = le.transform(test_df['clinical_diagnosis'])

    X_train, y_train = train_df[feature_cols].values, train_df['label'].values
    X_val, y_val = val_df[feature_cols].values, val_df['label'].values
    X_test, y_test = test_df[feature_cols].values, test_df['label'].values

    # 2. Preprocessing
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_p, X_val_p, X_test_p = scaler.fit_transform(imputer.fit_transform(X_train)), scaler.transform(imputer.transform(X_val)), scaler.transform(imputer.transform(X_test))

    # 3. DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_p, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_p, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_p, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader, val_loader, test_loader = DataLoader(train_ds, args.batch_size, shuffle=True), DataLoader(val_ds, args.batch_size), DataLoader(test_ds, args.batch_size)

    # 4. Model, Optimizer, Scheduler
    model = FlexibleMLP(X_train.shape[1], args.hidden_size, args.num_hidden_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_uar = 0.0
    patience_counter = 0
    save_path = os.path.join(run_results_dir, "best_model.pth")
    
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device)).squeeze(1)
            loss = criterion(outputs, y_batch.float().to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch.to(device)).squeeze(1)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())
        
        val_uar = balanced_accuracy_score(val_true, val_preds)
        scheduler.step(val_uar)

        if val_uar > best_val_uar:
            best_val_uar = val_uar
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0: print(f"  Ep {epoch+1}: Val UAR: {val_uar:.4f} (Best: {best_val_uar:.4f})")
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # 6. Final Test & Feature Importance
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device)).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(y_batch.numpy())
    
    report = classification_report(test_true, test_preds, target_names=args.classes)
    print("\nFinal Test Results:")
    print(report)
    
    with open(os.path.join(run_results_dir, 'test_report.txt'), 'w') as f:
        f.write(report)
        
    analyze_feature_importance(model, feature_cols)

if __name__ == '__main__':
    main()


"""
python src/train_word_hyperparams.py \
    --metadata "/home/aleph/embeddings_vs_classics/data/metadata_fugu_full_buffer0_emparejado_GLOBAL_additive.csv" \
    --features_csv "/home/aleph/embeddings_vs_classics/data/REDLAT_features.csv" \
    --num_hidden_layers 3 \
    --hidden_size 1024 \
    --dropout 0.3 \
    --lr 1e-5 \
    --weight_decay 1e-5

"""