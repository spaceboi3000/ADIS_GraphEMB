import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs('./plots', exist_ok=True)

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

# 1. Execution Time by Dataset and Dimension
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    plt.plot(dataset_data['dim'], dataset_data['total_time_s'], 
             marker='o', linewidth=2, markersize=8, label=dataset)
plt.xlabel('Embedding Dimension')
plt.ylabel('Total Time (s)')
plt.title('Total Execution Time by Dataset and Dimension')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Memory Usage Comparison
plt.subplot(2, 2, 2)
bar_width = 0.25
x_pos = np.arange(len(df['dataset'].unique()))
datasets = df['dataset'].unique()

for i, dim in enumerate([64, 128, 256]):
    dim_data = df[df['dim'] == dim]['peak_tracemalloc_mb']
    plt.bar(x_pos + i * bar_width, dim_data, bar_width, 
            label=f'Dim {dim}', alpha=0.8)

plt.xlabel('Dataset')
plt.ylabel('Peak Memory Usage (MB)')
plt.title('Peak Memory Usage by Dataset and Dimension')
plt.xticks(x_pos + bar_width, datasets)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Fit Time vs Number of Graphs
plt.subplot(2, 2, 3)
colors = {'MUTAG': 'red', 'ENZYMES': 'blue', 'IMDB-MULTI': 'green'}
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    plt.scatter(dataset_data['n_graphs'], dataset_data['fit_time_s'], 
                c=colors[dataset], s=100, label=dataset, alpha=0.7)
plt.xlabel('Number of Graphs')
plt.ylabel('Fit Time (s)')
plt.title('Fit Time vs Dataset Size')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Memory Efficiency (Peak Memory / Number of Graphs)
plt.subplot(2, 2, 4)
df['memory_efficiency'] = df['peak_tracemalloc_mb'] / df['n_graphs']
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    plt.plot(dataset_data['dim'], dataset_data['memory_efficiency'], 
             marker='s', linewidth=2, markersize=8, label=dataset)
plt.xlabel('Embedding Dimension')
plt.ylabel('Memory per Graph (MB/graph)')
plt.title('Memory Efficiency by Dataset')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/performance_analysis_grid.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Detailed Time Breakdown
plt.figure(figsize=(10, 6))
time_metrics = ['fit_time_s', 'embed_time_s']
datasets = df['dataset'].unique()

for i, dataset in enumerate(datasets):
    plt.subplot(1, 3, i+1)
    dataset_data = df[df['dataset'] == dataset]
    
    bottom = np.zeros(len(dataset_data))
    for j, metric in enumerate(time_metrics):
        plt.bar(range(len(dataset_data)), dataset_data[metric], 
                bottom=bottom, label=metric.replace('_s', '').replace('_', ' ').title())
        bottom += dataset_data[metric]
    
    plt.title(f'{dataset} Time Breakdown')
    plt.xlabel('Dimension')
    plt.ylabel('Time (s)')
    plt.xticks(range(len(dataset_data)), dataset_data['dim'])
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('./plots/time_breakdown_by_dataset.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. RSS Memory Analysis
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
x_pos = np.arange(len(df))
bar_width = 0.35

plt.bar(x_pos - bar_width/2, df['rss_before_mb'], bar_width, 
        label='RSS Before', alpha=0.7, color='lightblue')
plt.bar(x_pos + bar_width/2, df['rss_after_mb'], bar_width, 
        label='RSS After', alpha=0.7, color='lightcoral')

plt.xlabel('Experiments')
plt.ylabel('RSS Memory (MB)')
plt.title('RSS Memory: Before vs After Execution')
plt.legend()
plt.xticks(x_pos, [f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
           rotation=45)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(len(df)), df['peak_tracemalloc_mb'], 
        color='orange', alpha=0.7)
plt.xlabel('Experiments')
plt.ylabel('Peak Memory (MB)')
plt.title('Peak Memory Usage During Execution')
plt.xticks(range(len(df)), [f"{row['dataset']}\nDim{row['dim']}" for _, row in df.iterrows()], 
           rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./plots/memory_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Performance Heatmap
plt.figure(figsize=(10, 6))

# Create pivot tables for heatmaps
time_pivot = df.pivot(index='dataset', columns='dim', values='total_time_s')
memory_pivot = df.pivot(index='dataset', columns='dim', values='peak_tracemalloc_mb')

plt.subplot(1, 2, 1)
sns.heatmap(time_pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Time (s)'})
plt.title('Total Execution Time (s)')

plt.subplot(1, 2, 2)
sns.heatmap(memory_pivot, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Memory (MB)'})
plt.title('Peak Memory Usage (MB)')

plt.tight_layout()
plt.savefig('./plots/performance_heatmaps.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots have been saved to the './plots/' directory:")
print("1. performance_analysis_grid.png - Comprehensive overview")
print("2. time_breakdown_by_dataset.png - Detailed time analysis")
print("3. memory_analysis.png - RSS and peak memory usage")
print("4. performance_heatmaps.png - Heatmap visualization")