import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs('/c/Users/Nick/src/ADIS/scripts/embeddings_graph2vec/plots', exist_ok=True)

# Data from the CSV
data = {
    'dataset': ['MUTAG', 'MUTAG', 'MUTAG', 'ENZYMES', 'ENZYMES', 'ENZYMES', 'IMDB-MULTI', 'IMDB-MULTI', 'IMDB-MULTI'],
    'method': ['Graph2Vec'] * 9,
    'dim': [64, 128, 256, 64, 128, 256, 64, 128, 256],
    'n_graphs': [188, 188, 188, 600, 600, 600, 1500, 1500, 1500],
    'fit_time_s': [0.211, 0.1772, 0.1774, 0.9536, 0.9631, 1.2432, 1.4137, 1.5435, 1.481],
    'embed_time_s': [0.0001, 0.0001, 0.0001, 0.0003, 0.0003, 0.0003, 0.0008, 0.0008, 0.0008],
    'total_time_s': [0.2111, 0.1773, 0.1775, 0.9539, 0.9635, 1.2435, 1.4144, 1.5443, 1.4818],
    'rss_before_mb': [347.5, 350.48, 354.56, 373.12, 383.89, 388.58, 389.89, 401.61, 403.56],
    'rss_after_mb': [350.2, 350.89, 354.86, 378.89, 378.56, 335.61, 399.25, 401.09, 403.03],
    'peak_tracemalloc_mb': [1.61, 1.39, 1.39, 9.89, 7.7, 7.7, 10.71, 8.28, 9.38]
}

df = pd.DataFrame(data)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# 1. Time Comparison with Log Scale (Main Insight)
plt.figure(figsize=(14, 10))

# Plot 1: Time components with log scale
plt.subplot(2, 2, 1)
x_pos = np.arange(len(df))
bar_width = 0.8

# Since embed time is so small, we'll plot fit time and total time separately
fit_bars = plt.bar(x_pos, df['fit_time_s'], bar_width, 
                   label='Fit Time', alpha=0.7, color='royalblue')
embed_bars = plt.bar(x_pos, df['embed_time_s'], bar_width, 
                     bottom=df['fit_time_s'], label='Embed Time', alpha=0.7, color='coral')

plt.yscale('log')
plt.ylabel('Time (seconds, log scale)')
plt.title('Time Components: Fit vs Embed (Log Scale)')
plt.xticks(x_pos, [f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
           rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, which='both')

# Plot 2: Ratio of Fit Time to Embed Time
plt.subplot(2, 2, 2)
df['fit_embed_ratio'] = df['fit_time_s'] / df['embed_time_s']

colors = ['red' if ratio > 1000 else 'blue' for ratio in df['fit_embed_ratio']]
bars = plt.bar(range(len(df)), df['fit_embed_ratio'], color=colors, alpha=0.7)

plt.ylabel('Fit Time / Embed Time Ratio')
plt.title('Ratio of Fit Time to Embed Time\n(Red bars > 1000x)')
plt.xticks(range(len(df)), [f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
           rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}x', ha='center', va='bottom', fontsize=8)

# Plot 3: Absolute time values with log scale by dataset
plt.subplot(2, 2, 3)
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    
    # Plot fit time (dominant)
    plt.plot(dataset_data['dim'], dataset_data['fit_time_s'], 
             marker='o', linewidth=3, markersize=8, 
             label=f'{dataset} - Fit', linestyle='-')
    
    # Plot embed time (very small)
    plt.plot(dataset_data['dim'], dataset_data['embed_time_s'], 
             marker='s', linewidth=2, markersize=6, 
             label=f'{dataset} - Embed', linestyle='--', alpha=0.8)

plt.yscale('log')
plt.xlabel('Embedding Dimension')
plt.ylabel('Time (seconds, log scale)')
plt.title('Fit vs Embed Time by Dataset (Log Scale)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, which='both')

# Plot 4: Percentage breakdown
plt.subplot(2, 2, 4)
df['embed_time_percentage'] = (df['embed_time_s'] / df['total_time_s']) * 100

plt.bar(range(len(df)), df['embed_time_percentage'], 
        color='lightgreen', alpha=0.7, edgecolor='darkgreen')
plt.ylabel('Embed Time as % of Total Time')
plt.title('Embedding Time as Percentage of Total Time')
plt.xticks(range(len(df)), [f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
           rotation=45)
plt.grid(True, alpha=0.3)

# Add percentage labels
for i, percentage in enumerate(df['embed_time_percentage']):
    plt.text(i, percentage + 0.1, f'{percentage:.3f}%', 
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('./plots/time_analysis_log_scale.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Detailed Stacked Area Chart with Log Scale
plt.figure(figsize=(12, 8))

# Create a more detailed visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Stacked bar chart with log scale
ind = np.arange(len(df))
width = 0.8

p1 = ax1.bar(ind, df['fit_time_s'], width, label='Fit Time', color='navy', alpha=0.8)
p2 = ax1.bar(ind, df['embed_time_s'], width, bottom=df['fit_time_s'], 
             label='Embed Time', color='red', alpha=0.8)

ax1.set_ylabel('Time (seconds, log scale)')
ax1.set_title('Stacked Time Components (Log Scale)\nEmbed Time Barely Visible!')
ax1.set_xticks(ind)
ax1.set_xticklabels([f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
                    rotation=45)
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Right: Just embed time (linear scale to see the actual values)
ax2.bar(ind, df['embed_time_s'], width, color='red', alpha=0.8)
ax2.set_ylabel('Embed Time (seconds, linear scale)')
ax2.set_title('Embed Time Only (Linear Scale)')
ax2.set_xticks(ind)
ax2.set_xticklabels([f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
                    rotation=45)

# Add value labels on embed time bars
for i, v in enumerate(df['embed_time_s']):
    ax2.text(i, v + 0.0001, f'{v:.4f}s', ha='center', va='bottom', fontsize=8)

ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/time_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Time Ratios Heatmap
plt.figure(figsize=(10, 6))

# Calculate various ratios
df['fit_embed_ratio'] = df['fit_time_s'] / df['embed_time_s']
df['embed_percentage'] = (df['embed_time_s'] / df['total_time_s']) * 100

# Create pivot tables
ratio_pivot = df.pivot(index='dataset', columns='dim', values='fit_embed_ratio')
percentage_pivot = df.pivot(index='dataset', columns='dim', values='embed_percentage')

plt.subplot(1, 2, 1)
sns.heatmap(ratio_pivot, annot=True, fmt='.0f', cmap='Reds', 
            cbar_kws={'label': 'Fit Time / Embed Time Ratio'})
plt.title('Fit Time to Embed Time Ratio\n(Higher = more disparity)')

plt.subplot(1, 2, 2)
sns.heatmap(percentage_pivot, annot=True, fmt='.3f', cmap='Blues',
            cbar_kws={'label': 'Embed Time % of Total'})
plt.title('Embed Time as % of Total Time')

plt.tight_layout()
plt.savefig('./plots/time_ratios_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Magnitude Comparison
plt.figure(figsize=(10, 6))

# Calculate orders of magnitude difference
df['magnitude_difference'] = np.log10(df['fit_time_s'] / df['embed_time_s'])

plt.bar(range(len(df)), df['magnitude_difference'], 
        color='purple', alpha=0.7, edgecolor='darkviolet')

plt.ylabel('Orders of Magnitude Difference\n(log10(Fit Time / Embed Time))')
plt.title('Orders of Magnitude Difference Between Fit and Embed Times')
plt.xticks(range(len(df)), [f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
           rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels
for i, mag in enumerate(df['magnitude_difference']):
    plt.text(i, mag + 0.05, f'10^{mag:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('./plots/magnitude_difference.png', dpi=300, bbox_inches='tight')
plt.close()

print("All log-scale plots have been saved to the './plots/' directory:")
print("1. time_analysis_log_scale.png - Comprehensive log-scale analysis")
print("2. time_comparison_detailed.png - Stacked bars with log scale")
print("3. time_ratios_heatmap.png - Ratio heatmaps")
print("4. magnitude_difference.png - Orders of magnitude difference")
print("\nKey Insights:")
print(f"- Fit time is {df['fit_embed_ratio'].min():.0f}x to {df['fit_embed_ratio'].max():.0f}x larger than embed time")
print(f"- Embed time represents only {df['embed_percentage'].min():.4f}% to {df['embed_percentage'].max():.4f}% of total time")
print(f"- Fit time dominates by 3-4 orders of magnitude (10^{df['magnitude_difference'].min():.1f} to 10^{df['magnitude_difference'].max():.1f})")