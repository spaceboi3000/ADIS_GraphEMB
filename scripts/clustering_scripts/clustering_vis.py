import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("clustering_merged.csv")
print(df.info())
print(df.head())
print("Unique datasets:", df['dataset'].unique())
print("Unique methods:", df['method'].unique())
print("Unique dims:", df['dim'].unique())
print("Unique cluster_alg:", df['cluster_alg'].unique())

#dimension consistency?
methods = df['method'].unique()
for m in methods:
    print(f"Method: {m}")
    for d in df[df['method'] == m]['dataset'].unique():
        dims = df[(df['method'] == m) & (df['dataset'] == d)]['dim'].unique()
        print(f"  Dataset: {d}, Dims: {sorted(dims)}")


metrics = ['nmi', 'ari', 'acc', 'silhouette']
methods = df['method'].unique()
cluster_algs = ['kmeans', 'spectral']
datasets = df['dataset'].unique()
colors = {'ENZYMES': 'blue', 'IMDB-MULTI': 'green', 'MUTAG': 'red'}


#we rank dims within each (method, dataset) group to identify Low(1), Mid(2), High(3)
df['dim_rank'] = df.groupby(['method', 'dataset'])['dim'].rank(method='dense').astype(int)

def generate_plots_v2():
    plot_files = []
    for alg in cluster_algs:
        for method in methods:
            
            #filter for Method and Alg
            mask = (df['method'] == method) & (df['cluster_alg'] == alg)
            data_subset = df[mask].copy()
            
            if data_subset.empty:
                continue
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Method: {method} | Clustering: {alg}", fontsize=18)
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                
                # best score for each dataset+dim combination
                best_scores = data_subset.groupby(['dataset', 'dim', 'dim_rank'])[metric].max().reset_index()
                
                for ds in datasets:
                    ds_data = best_scores[best_scores['dataset'] == ds].sort_values('dim')
                    if not ds_data.empty:
                        ax.plot(ds_data['dim'], ds_data[metric], marker='o', label=ds, color=colors.get(ds, 'gray'))
    
                #average the Metric and the Dimension for the points
                avg_data = best_scores.groupby('dim_rank').agg({metric: 'mean','dim': 'mean'}).reset_index().sort_values('dim_rank')
                
                ax.plot(avg_data['dim'], avg_data[metric], marker='s', label='Average', color='black', linewidth=2, linestyle='--')
                ax.set_title(metric.upper())
                ax.set_ylabel(metric)
                ax.set_xlabel('Embedding Dimension')
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # For GIN, ensure we see all ticks if they aren't too crowded
                if method == 'GIN':
                     # Collect all unique dims for x-ticks
                     unique_dims = sorted(best_scores['dim'].unique())
                     ax.set_xticks(unique_dims)
                     ax.set_xticklabels(unique_dims, rotation=45)
                else:
                    # For others, dims are usually 64, 128, 256
                    unique_dims = sorted(best_scores['dim'].unique())
                    ax.set_xticks(unique_dims)
                if idx == 0:
                    ax.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = f"{method}_{alg}_performance.png"
            plt.savefig(filename)
            plot_files.append(filename)
            plt.close()
    return plot_files

created_files = generate_plots_v2()
print(created_files)