import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
from pathlib import Path

# --- HELPER: Prettify Task Names ---
def get_task_label(task_list):
    """Converts raw task list to a clean label."""
    if not task_list: return "Unknown"
    
    # Define known sets
    all_tasks = set(['Fugu', 'CraftIm', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2'])
    curr_tasks = set(task_list)
    
    # Check for exact matches
    if curr_tasks == set(['Fugu', 'CraftIm', 'Phonological', 'Phonological2', 'Semantic', 'Semantic2']): return "All Tasks"
    if curr_tasks == set(['Fugu']): return "Fugu Only"
    if curr_tasks == set(['CraftIm']): return "CraftIm Only"
    if curr_tasks.issuperset({'Phonological', 'Semantic'}): return "Fluency Tasks"
    if len(curr_tasks) > 4: return "All Tasks"
    
    return ", ".join(task_list)

def get_pretty_name(row):
    """Maps raw config to paper-ready model names."""
    m_type = row['model_type']
    subset = row.get('classic_subset', None)
    
    if m_type == 'classic':
        if subset == 'audio': return 'Classic Audio'
        if subset == 'language': return 'Classic Lang'
        return 'Classic Combined'
    elif m_type == 'classic_fusion':
        return 'Classic Fusion (GMU)'
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
    print(f"🔍 Found {len(config_files)} experiments...")

    for cf in config_files:
        exp_path = Path(cf).parent
        try:
            with open(cf, 'r') as f:
                config = json.load(f)
            
            res_file = exp_path / "results.csv"
            if not res_file.exists(): continue
            
            df = pd.read_csv(res_file)
            
            # Basic Stats
            mean_auc = df['auc'].mean()
            std_auc = df.groupby('seed')['auc'].mean().std()
            
            # Tasks Label
            task_label = get_task_label(config.get('tasks', []))

            entry = {
                'path': str(exp_path),
                'Tasks': task_label,  # <--- NEW FIELD
                'group_pair': " vs ".join(config['target_groups']),
                'model_type': config['model_type'],
                'classic_subset': config.get('classic_subset', None),
                'auc': mean_auc,
                'std': std_auc if not np.isnan(std_auc) else 0.0,
                'ci_lower': df['ci_lower'].mean(),
                'ci_upper': df['ci_upper'].mean()
            }
            
            # Check for Weights/Gates
            if 'gate_audio_trust' in df.columns:
                entry['gate_audio'] = df['gate_audio_trust'].mean()
            
            layer_cols = [c for c in df.columns if c.startswith('L') and c[1:].isdigit()]
            if layer_cols:
                avg_weights = df[layer_cols].mean().to_dict()
                entry.update(avg_weights)
                entry['has_weights'] = True
            else:
                entry['has_weights'] = False
                
            data.append(entry)
            
        except Exception as e:
            print(f"Skipping {exp_path}: {e}")

    df_all = pd.DataFrame(data)
    if df_all.empty: return pd.DataFrame()

    df_all['Model'] = df_all.apply(get_pretty_name, axis=1)
    
    # Sort Order
    model_order = ['Classic Audio', 'Classic Lang', 'Classic Combined', 'Classic Fusion (GMU)', 'WavLM', 'RoBERTa', 'Fusion (GMU)']
    df_all['Model'] = pd.Categorical(df_all['Model'], categories=model_order, ordered=True)
    
    return df_all.sort_values(['Tasks', 'group_pair', 'Model'])

def plot_bar_chart(df, task_name):
    """Generates Bar Chart for a specific task set"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    g = sns.barplot(
        data=df, 
        x='group_pair', 
        y='auc', 
        hue='Model', 
        palette='viridis',
        edgecolor='black'
    )
    
    plt.ylim(0.5, 1.0)
    plt.title(f"Performance: {task_name}", fontsize=16)
    plt.ylabel("Mean AUC", fontsize=14)
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fname = f"results_{task_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"📊 Saved '{fname}'")
    plt.close()

def plot_wavlm_weights(df, task_name):
    """Plots WavLM weights for a specific task set"""
    wav_df = df[(df['model_type'] == 'wavlm') & (df['has_weights'] == True)]
    if wav_df.empty: return

    plt.figure(figsize=(10, 5))
    layer_cols = [f'L{i}' for i in range(13)]
    
    for _, row in wav_df.iterrows():
        weights = [row[c] for c in layer_cols]
        plt.plot(range(13), weights, marker='o', label=row['group_pair'], linewidth=2)

    plt.title(f"WavLM Layer Weights ({task_name})", fontsize=16)
    plt.xlabel("Layer (0=Bottom, 12=Top)", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.xticks(range(13))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    fname = f"weights_{task_name.replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    print(f"📈 Saved '{fname}'")
    plt.close()

if __name__ == "__main__":
    df = load_experiments()
    
    if not df.empty:
        # Define Columns
        cols = ['Tasks', 'group_pair', 'Model', 'auc', 'ci_lower', 'ci_upper', 'std']
        if 'gate_audio' in df.columns: cols.append('gate_audio')
        
        # 1. Save Full CSV
        df[cols].to_csv("final_results_summary.csv", index=False)
        print(f"\n💾 Saved full summary to: final_results_summary.csv")
        
        # 2. Loop over each Task Set found
        unique_tasks = df['Tasks'].unique()
        print(f"\nFound Task Sets: {unique_tasks}")
        
        for task in unique_tasks:
            print(f"\n=== Report for: {task} ===")
            sub_df = df[df['Tasks'] == task]
            
            # Print Table
            print(sub_df[['group_pair', 'Model', 'auc', 'ci_lower', 'ci_upper']].to_string(index=False))
            
            # Generate Plots per Task
            plot_bar_chart(sub_df, task)
            plot_wavlm_weights(sub_df, task)