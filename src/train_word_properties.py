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
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# ================= CONFIGURATION =================
DEFAULT_TASKS = ['CraftIm', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2', 'Fugu']
DEFAULT_FEATURES = ["concreteness", "granularity", "mean_num_syll", "mean_phon_neigh", "mean_log_frq", "pitch", "timing"]
# =================================================

class LinguisticMLP(nn.Module):
    def __init__(self, input_dim, hidden_size=1024, dropout_rate=0.5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )
        self.input_layer = self.network[0]

    def forward(self, x):
        return self.network(x)

def analyze_feature_importance(model, feature_names, n_top=20):
    """
    Analyzes the absolute weights of the first layer to find the most important features.
    """
    print("\n" + "="*40)
    print("=== FEATURE IMPORTANCE ANALYSIS ===")
    print("="*40)

    # --- START FIX ---
    # Get the weight matrix from the first linear layer. Shape: [hidden_size, input_dim]
    weights_matrix = model.input_layer.weight.data.cpu().numpy()
    
    # To get a single score per input feature, we sum the absolute values of its
    # outgoing weights across all neurons in the first hidden layer.
    # We sum along axis 0.
    importance_scores = np.sum(np.abs(weights_matrix), axis=0)
    
    # --- END FIX ---
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print(f"Top {n_top} most important linguistic features:\n")
    print(importance_df.head(n_top).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on linguistic features from separate files.")
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--features_csv', type=str, required=True)
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--features', nargs='+', default=DEFAULT_FEATURES)
    parser.add_argument('--classes', nargs='+', default=['AD', 'CN'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1. Load and Merge Data
    meta_df = pd.read_csv(args.metadata, low_memory=False)
    features_df = pd.read_csv(args.features_csv, low_memory=False)
    df = pd.merge(meta_df[['record_id', 'clinical_diagnosis']], features_df, on='record_id', how='inner')
    
    # 2. Select Features
    feature_cols = [col for col in df.columns if any(task in col for task in args.tasks) and any(feat in col for feat in args.features)]
    print(f"Found {len(feature_cols)} linguistic feature columns to use.")

    # 3. Patient-level Stratified Split
    df = df.dropna(subset=['clinical_diagnosis'])
    df = df[df['clinical_diagnosis'].isin(args.classes)]
    unique_patients = df[['record_id', 'clinical_diagnosis']].drop_duplicates()
    
    X_ids = unique_patients['record_id'].values
    y_labels_unique = unique_patients['clinical_diagnosis'].values
    
    train_val_ids, test_ids, y_train_val_labels, _ = train_test_split(
        X_ids, y_labels_unique, test_size=0.15, stratify=y_labels_unique, random_state=42
    )
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids, y_train_val_labels, test_size=0.1765, stratify=y_train_val_labels, random_state=42
    )

    train_df = df[df['record_id'].isin(train_ids)].copy()
    val_df = df[df['record_id'].isin(val_ids)].copy()
    test_df = df[df['record_id'].isin(test_ids)].copy()
    
    print(f"Split sizes (patients): Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 4. Prepare Feature Matrices and Labels
    le = LabelEncoder()
    le.fit(df['clinical_diagnosis'])
    
    train_df['label_encoded'] = le.transform(train_df['clinical_diagnosis'])
    val_df['label_encoded'] = le.transform(val_df['clinical_diagnosis'])
    test_df['label_encoded'] = le.transform(test_df['clinical_diagnosis'])

    X_train = train_df[feature_cols].values
    y_train = train_df['label_encoded'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label_encoded'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label_encoded'].values

    # 5. Preprocessing (Impute & Scale)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
    X_val_processed = scaler.transform(imputer.transform(X_val))
    X_test_processed = scaler.transform(imputer.transform(X_test))

    # 6. DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_processed, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_processed, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_processed, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 7. Model Training
    model = LinguisticMLP(X_train.shape[1], args.hidden_size, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_val_uar = 0.0
    patience_counter = 0

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

        if val_uar > best_val_uar:
            best_val_uar = val_uar
            torch.save(model.state_dict(), "best_linguistic_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1}: Val UAR: {val_uar:.4f} (Best: {best_val_uar:.4f})")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # 8. Final Test Evaluation
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load("best_linguistic_model.pth"))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device)).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(y_batch.numpy())
    
    print("\nFinal Test Results:")
    print(classification_report(test_true, test_preds, target_names=args.classes))
    
    # 9. Interpretability
    analyze_feature_importance(model, feature_cols)

if __name__ == '__main__':
    main()


"""
python src/train_word_properties.py \
    --metadata "/home/aleph/embeddings_vs_classics/data/metadata_fugu_full_buffer0_emparejado_GLOBAL_additive.csv" \
    --features_csv "/home/aleph/embeddings_vs_classics/data/REDLAT_features.csv" \
    --classes AD CN
"""