import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from data_manager import AUDIO_KEYS, LANG_KEYS  # Import your definitions

def get_modality(feature_name):
    """Classifies a feature name into Audio or Language based on keys."""
    if any(k in feature_name for k in AUDIO_KEYS):
        return 'Audio'
    if any(k in feature_name for k in LANG_KEYS):
        return 'Language'
    return 'Other'

def calculate_permutation_importance(model, X_val, y_val, feature_names, device, metric='auc'):
    """
    Calculates importance by shuffling one feature at a time and measuring performance drop.
    """
    model.eval()
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    # y_val should be numpy for sklearn metric
    
    # 1. Get Baseline Performance
    with torch.no_grad():
        baseline_preds = model(X_val_tensor).cpu().numpy().flatten()
    
    baseline_score = roc_auc_score(y_val, baseline_preds)
    
    importances = []
    
    # 2. Iterate over every feature
    print(f"Calculating importance for {len(feature_names)} features...")
    
    for i, col_name in enumerate(feature_names):
        # Create a copy of the tensor to corrupt
        X_corrupted = X_val_tensor.clone()
        
        # Shuffle the i-th feature column
        # We do this on GPU/Tensor directly for speed
        idx = torch.randperm(X_corrupted.shape[0])
        X_corrupted[:, i] = X_corrupted[idx, i]
        
        # Predict
        with torch.no_grad():
            preds = model(X_corrupted).cpu().numpy().flatten()
            
        score = roc_auc_score(y_val, preds)
        
        # Importance = Drop in performance
        # (Higher positive value = Feature is more important)
        importance_val = baseline_score - score
        
        importances.append({
            'Feature': col_name,
            'Importance': importance_val,
            'Modality': get_modality(col_name)
        })

    # Create DataFrame
    df_imp = pd.DataFrame(importances)
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    return df_imp

def plot_top_features(df_imp, top_n=15, title="Top Feature Importance (Permutation)"):
    """
    Plots a beautiful horizontal bar chart grouped by modality.
    """
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Take top N
    subset = df_imp.head(top_n)
    
    # Define colors
    palette = {'Audio': '#1f77b4', 'Language': '#ff7f0e', 'Other': '#7f7f7f'}
    
    # Plot
    sns.barplot(
        data=subset,
        y='Feature',
        x='Importance',
        hue='Modality',
        dodge=False,
        palette=palette
    )
    
    plt.title(title, fontsize=15)
    plt.xlabel("Drop in AUC (Importance)", fontsize=12)
    plt.ylabel("")
    plt.legend(title='Modality', loc='lower right')
    plt.tight_layout()
    
    # Save or Show
    plt.savefig(f"feature_importance_top{top_n}.png", dpi=300)
    print(f"Plot saved to feature_importance_top{top_n}.png")
    plt.show()
