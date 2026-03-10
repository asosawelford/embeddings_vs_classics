import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from data_manager import AUDIO_KEYS, LANG_KEYS

def get_modality(feature_name):
    """Classifies a feature name into Audio or Language based on keys."""
    if any(k in feature_name for k in AUDIO_KEYS):
        return 'Audio'
    if any(k in feature_name for k in LANG_KEYS):
        return 'Language'
    return 'Audio'

def get_feature_family(feature_name):
    """
    Extracts Task and Family from strings like:
    'Semantic2__concreteness__concreteness_median_concr'
    Returns: 'Semantic2__concreteness'
    """
    parts = feature_name.split('__')
    if len(parts) >= 2:
        return f"{parts[0]}__{parts[1]}"
    return feature_name # Fallback if no '__' is found

def calculate_permutation_importance(model, X_val, y_val, feature_names, device, group_by_family=True):
    """
    Calculates importance by shuffling one feature at a time and measuring performance drop.
    Optionally groups by feature family.
    """
    model.eval()
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    
    # 1. Get Baseline Performance
    with torch.no_grad():
        baseline_preds = model(X_val_tensor).cpu().numpy().flatten()
    
    baseline_score = roc_auc_score(y_val, baseline_preds)
    importances = []
    
    # 2. Iterate over every single feature
    print(f"Calculating importance for {len(feature_names)} features...")
    
    for i, col_name in enumerate(feature_names):
        X_corrupted = X_val_tensor.clone()
        
        # Shuffle the i-th feature column
        idx = torch.randperm(X_corrupted.shape[0])
        X_corrupted[:, i] = X_corrupted[idx, i]
        
        # Predict
        with torch.no_grad():
            preds = model(X_corrupted).cpu().numpy().flatten()
            
        score = roc_auc_score(y_val, preds)
        
        # Drop in performance
        importance_val = baseline_score - score
        
        importances.append({
            'Feature': col_name,
            'Importance': importance_val,
            'Modality': get_modality(col_name)
        })

    # 3. Create DataFrame
    df_imp = pd.DataFrame(importances)
    
    # 4. Group by Family if requested
    if group_by_family:
        print("Grouping importances by Feature Family...")
        # Add a family column
        df_imp['Family'] = df_imp['Feature'].apply(get_feature_family)
        
        # Group by Family + Modality, and SUM the importances
        df_grouped = df_imp.groupby(['Family', 'Modality'], as_index=False)['Importance'].sum()
        
        # Rename 'Family' back to 'Feature' for the plotting function
        df_grouped = df_grouped.rename(columns={'Family': 'Feature'})
        df_imp = df_grouped

    # 5. Sort descending
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    
    return df_imp

def plot_top_features(df_imp, top_n=15, title="Top Feature Importance (Grouped)"):
    """
    Plots a beautiful horizontal bar chart grouped by modality.
    """
    plt.figure(figsize=(12, 8)) # Made slightly wider to fit longer Task__Family names
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
    plt.xlabel("Total Drop in AUC (Combined Importance)", fontsize=12)
    plt.ylabel("")
    plt.legend(title='Modality', loc='lower right')
    plt.tight_layout()
    
    # Save or Show
    # Use a safe filename without spaces or weird chars
    safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    filename = f"{safe_title}.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")
    # plt.show() # Uncomment if running in a notebook