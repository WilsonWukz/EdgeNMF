import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

file_path = 'train/ALL_ALGO_SUMMARY.xlsx'
df = pd.read_excel(file_path)

def extract_p(s):
    if isinstance(s, str):
        match = re.search(r"([\d\.]+)", s)
        return float(match.group(1)) if match else 0.0
    return s

if '(p,r)' in df.columns:
    df['Noise_Level'] = df['(p,r)'].apply(extract_p)
else:
    df['Noise_Level'] = df.iloc[:, 0].apply(extract_p)

df_orl = df[df['Data'] == 'ORL'].copy()

numeric_cols = ['ACC_mean', 'ACC_std', 'Time_mean', 'Time_std', 'Memory_mean', 'Memory_std']
for col in numeric_cols:
    df_orl[col] = pd.to_numeric(df_orl[col], errors='coerce')

sns.set(style="whitegrid", context="paper", font_scale=1.4)
palette = {
    "L1": "#2ca02c",
    "L2": "#1f77b4",
    "MU": "#ff7f0e",
    "HC": "#9467bd",
    "StackedNMF": "#d62728",
    "StackMU": "#d62728"
}

# Fig 1: Robustness Analysis (ACC vs Noise)
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df_orl,
    x='Noise_Level',
    y='ACC_mean',
    hue='Algo',
    style='Algo',
    palette=palette,
    markers=True,
    dashes=False,
    linewidth=3,
    markersize=10
)

plt.axvspan(0.1, 0.3, color='green', alpha=0.1, label='Moderate Noise Region')

plt.title('Robustness Analysis: Accuracy Degradation under Noise', fontsize=16, fontweight='bold')
plt.xlabel('Impulsive Noise Ratio ($p$)', fontsize=14)
plt.ylabel('Clustering Accuracy (ACC)', fontsize=14)
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Fig1_Robustness.png', dpi=300)
plt.show()

# Fig 2: Computational Efficiency (Time & Memory)
efficiency_df = df_orl.groupby('Algo')[['Time_mean', 'Memory_mean']].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(
    data=efficiency_df,
    x='Algo',
    y='Time_mean',
    palette=palette,
    ax=axes[0],
    edgecolor='black'
)
axes[0].set_title('Average Inference Time (s)', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Time (seconds)', fontsize=14)
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(
    data=efficiency_df,
    x='Algo',
    y='Memory_mean',
    palette=palette,
    ax=axes[1],
    edgecolor='black'
)
axes[1].set_title('Peak Memory Usage (MB)', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Memory (MB)', fontsize=14)
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Fig2_Efficiency.png', dpi=300)
plt.show()

# Fig 3: Efficiency-Robustness Trade-off (at p=0.2)
target_p = 0.2
tradeoff_df = df_orl[df_orl['Noise_Level'] == target_p].copy()

plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=tradeoff_df,
    x='Time_mean',
    y='ACC_mean',
    hue='Algo',
    style='Algo',
    palette=palette,
    s=300,
    alpha=0.9,
    edgecolor='black'
)

for i in range(tradeoff_df.shape[0]):
    row = tradeoff_df.iloc[i]
    plt.text(
        row['Time_mean'] + 0.02,
        row['ACC_mean'] + 0.005,
        f"{row['Algo']}",
        fontsize=12,
        fontweight='bold'
    )

plt.gca().text(
    0.05, 0.95,
    "Edge Sweet Spot\n(High ACC, Low Time)",
    transform=plt.gca().transAxes,
    fontsize=14,
    color='green',
    verticalalignment='top',
    bbox=dict(boxstyle="round", fc="white", ec="green", alpha=0.8)
)

plt.title(f'Efficiency-Robustness Trade-off (at Noise $p={target_p}$)', fontsize=16, fontweight='bold')
plt.xlabel('Computational Cost (Inference Time [s])', fontsize=14)
plt.ylabel(f'Robustness (Accuracy at $p={target_p}$)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Fig3_Tradeoff.png', dpi=300)
plt.show()