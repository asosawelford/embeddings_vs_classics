import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path

def get_pretty_name(row):
    """Maps raw config to paper-ready model names."""
    m_type = row['model_type']
    subset = row.get('classic_subset', None)
    
    if m_type == 'classic':
        if subset == 'audio': return 'Classic Audio'
        if subset == 'language': return 'Classic Lang'
        return 'Classic Combined'
    elif m_type == 'wavlm':
        return 'WavLM'
    elif m_type == 'roberta':
        return 'RoBERTa'
    elif m_type == 'fusion':
        return 'Fusion (GMU)'
    return m_type.capitalize()

def load_experiments(exp_dir="experiments"):
    data = []
    
    # Find all config.json files
    config_files = glob.glob(f"{exp_dir}/*/config.json")
    
    print(f"found {len(config_files)} experiments...")

    for cf in config_files:
        exp_path = Path(cf).parent
        try:
            # Load Config
            with open(cf, 'r') as f:
                config = json.load(f)
            
            # Load Results
            res_file = exp_path / "results.csv"
            if not res_file.exists():
                continue
            
            df = pd.read_csv(res_file)
            
            # Aggregate Results (Mean over seeds/folds)
            mean_auc = df['auc'].mean()
            std_auc = df.groupby('seed')['auc'].mean().std()
            
            # CI is tricky to aggregate, so we take the mean of the bounds
            mean_lower = df['ci_lower'].mean()
            mean_upper = df['ci_upper'].mean()
            
            # Create Record
            entry = {
                'path': str(exp_path),
                'group_pair': " vs ".join(config['target_groups']),
                'model_type': config['model_type'],
                'classic_subset': config.get('classic_subset', None),
                'auc': mean_auc,
                'std': std_auc if not np.isnan(std_auc) else 0.0,
                'ci_lower': mean_lower,
                'ci_upper': mean_upper
            }
            
            # Check for Layer Weights (L0...L12)
            layer_cols = [c for c in df.columns if c.startswith('L') and c[1:].isdigit()]
            if layer_cols:
                # Average weights across all folds
                avg_weights = df[layer_cols].mean().to_dict()
                entry.update(avg_weights)
                entry['has_weights'] = True
            else:
                entry['has_weights'] = False
                
            data.append(entry)
            
        except Exception as e:
            print(f"Skipping {exp_path}: {e}")

    # Convert to DataFrame
    df_all = pd.DataFrame(data)
    if df_all.empty:
        print("No results found!")
        return pd.DataFrame()

    # Create Pretty Name
    df_all['Model'] = df_all.apply(get_pretty_name, axis=1)
    
    # Sort order
    model_order = ['Classic Audio', 'Classic Lang', 'Classic Combined', 'WavLM', 'RoBERTa', 'Fusion (GMU)']
    df_all['Model'] = pd.Categorical(df_all['Model'], categories=model_order, ordered=True)
    
    return df_all.sort_values(['group_pair', 'Model'])

def plot_bar_chart(df):
    """Generates the Main Result Bar Chart"""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create the plot
    g = sns.barplot(
        data=df, 
        x='group_pair', 
        y='auc', 
        hue='Model', 
        palette='viridis',
        edgecolor='black'
    )
    
    # Add Error Bars (Approximated by CI distance)
    # Note: Seaborn aggregates raw data, but here we have pre-aggregated means.
    # We will manually add error bars using the CI columns.
    
    # Iterate over patches to place error bars
    # This is complex in Seaborn, so simplified standard plot:
    
    plt.ylim(0.5, 1.0)
    plt.title("Classification Performance (AUC) by Modality", fontsize=16)
    plt.ylabel("Mean AUC", fontsize=14)
    plt.xlabel("Comparison Group", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=300)
    print("📊 Saved 'results_comparison.png'")
    plt.show()

def plot_wavlm_weights(df):
    """Plots the learned layer weights for WavLM models"""
    # Filter only WavLM entries that have weights
    wav_df = df[(df['model_type'] == 'wavlm') & (df['has_weights'] == True)]
    
    if wav_df.empty:
        return

    plt.figure(figsize=(10, 5))
    
    # Prepare data for plotting
    layer_cols = [f'L{i}' for i in range(13)]
    
    # Plot a line for each group comparison
    for _, row in wav_df.iterrows():
        weights = [row[c] for c in layer_cols]
        plt.plot(range(13), weights, marker='o', label=row['group_pair'], linewidth=2)

    plt.title("WavLM Learnable Layer Weights (Importance)", fontsize=16)
    plt.xlabel("Layer Depth (0=CNN, 12=Top Transformer)", fontsize=14)
    plt.ylabel("Softmax Weight", fontsize=14)
    plt.xticks(range(13))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("wavlm_weights.png", dpi=300)
    print("📈 Saved 'wavlm_weights.png'")
    plt.show()

def print_latex_table(df):
    """Prints a LaTeX formatted table for the paper"""
    print("\n\n--- LaTeX Table Snippet ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l|ccc}")
    print("\\hline")
    print("Model & CN vs AD & CN vs FTD & AD vs FTD \\\\")
    print("\\hline")
    
    models = df['Model'].unique().sort_values()
    groups = df['group_pair'].unique()
    
    for m in models:
        row_str = f"{m}"
        for g in groups:
            # Find the row
            match = df[(df['Model'] == m) & (df['group_pair'] == g)]
            if not match.empty:
                auc = match.iloc[0]['auc']
                ci_low = match.iloc[0]['ci_lower']
                ci_high = match.iloc[0]['ci_upper']
                row_str += f" & {auc:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
            else:
                row_str += " & -"
        row_str += " \\\\"
        print(row_str)
        
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{AUC scores with 95\\% Confidence Intervals.}")
    print("\\end{table}")
    print("---------------------------\n")

if __name__ == "__main__":
    df = load_experiments()
    
    if not df.empty:
        # Define columns to show in summary
        summary_cols = ['group_pair', 'Model', 'auc', 'ci_lower', 'ci_upper', 'std']
        
        # 1. Print to Terminal
        print("\n=== SUMMARY TABLE ===")
        print(df[summary_cols].to_string(index=False))
        
        # 2. Save to CSV (NEW)
        csv_filename = "final_results_summary.csv"
        df[summary_cols].to_csv(csv_filename, index=False)
        print(f"\n💾 Summary saved to: {csv_filename}")
        
        # 3. Generate Plots & Latex
        plot_bar_chart(df)
        plot_wavlm_weights(df)
        # print_latex_table(df)