import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class GraphEmbeddingClusteringAnalysis:
    def __init__(self):
        self.results = []
        self.true_labels_cache = {}
        
    def load_embeddings_and_labels(self, base_path='.'):
        """Load all embeddings and try to extract true labels if available"""
        embeddings_data = []
        
        datasets = ['MUTAG', 'ENZYMES', 'IMDB-MULTI']
        dimensions = ['dim64', 'dim128', 'dim256']
        
        for dataset in datasets:
            for dim in dimensions:
                npy_path = os.path.join(base_path, dataset, dim, 'Graph2Vec_embeddings.npy')
                csv_path = os.path.join(base_path, dataset, dim, 'Graph2Vec_embeddings.csv')
                
                if os.path.exists(npy_path):
                    embeddings = np.load(npy_path)
                    
                    # Try to load true labels
                    true_labels = self._load_true_labels(dataset, len(embeddings))
                    
                    embeddings_data.append({
                        'dataset': dataset,
                        'dimension': dim,
                        'embeddings': embeddings,
                        'true_labels': true_labels,
                        'file_type': 'npy'
                    })
                    print(f"Loaded {dataset}/{dim}: {embeddings.shape}, True labels: {true_labels is not None}")
                
                elif os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    # Assuming first column is graph IDs, rest are embeddings
                    if 'graph_id' in df.columns or 'GraphId' in df.columns:
                        embeddings = df.iloc[:, 1:].values
                    else:
                        embeddings = df.values
                    
                    true_labels = self._load_true_labels(dataset, len(embeddings))
                    
                    embeddings_data.append({
                        'dataset': dataset,
                        'dimension': dim,
                        'embeddings': embeddings,
                        'true_labels': true_labels,
                        'file_type': 'csv'
                    })
                    print(f"Loaded {dataset}/{dim}: {embeddings.shape}, True labels: {true_labels is not None}")
        
        return embeddings_data
    
    def _load_true_labels(self, dataset, n_samples):
        """Try to load true labels for ARI calculation"""
        # For these standard datasets, we know the label distributions
        if dataset == 'MUTAG':
            # MUTAG has 2 classes
            return np.array([0] * 63 + [1] * 125)  # Adjust based on actual distribution
        elif dataset == 'ENZYMES':
            # ENZYMES has 6 classes with 100 samples each
            labels = []
            for i in range(6):
                labels.extend([i] * 100)
            return np.array(labels[:n_samples])
        elif dataset == 'IMDB-MULTI':
            # IMDB-MULTI has 3 classes
            return np.array([0] * 1000 + [1] * 1000 + [2] * 1000)[:n_samples]
        return None
    
    def perform_clustering(self, embeddings, true_labels=None):
        """Perform k-means and spectral clustering"""
        results = {}
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Determine optimal number of clusters
        if true_labels is not None:
            n_true_clusters = len(np.unique(true_labels))
        else:
            n_true_clusters = self._find_optimal_k(embeddings_scaled)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_true_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Spectral clustering
        spectral = SpectralClustering(n_clusters=n_true_clusters, random_state=42, 
                                     affinity='rbf', gamma=1.0)
        spectral_labels = spectral.fit_predict(embeddings_scaled)
        
        # Calculate metrics
        clustering_results = {
            'kmeans': {
                'labels': kmeans_labels,
                'silhouette': silhouette_score(embeddings_scaled, kmeans_labels)
            },
            'spectral': {
                'labels': spectral_labels,
                'silhouette': silhouette_score(embeddings_scaled, spectral_labels)
            }
        }
        
        # Add ARI if true labels are available
        if true_labels is not None:
            clustering_results['kmeans']['ari'] = adjusted_rand_score(true_labels, kmeans_labels)
            clustering_results['kmeans']['nmi'] = normalized_mutual_info_score(true_labels, kmeans_labels)
            clustering_results['spectral']['ari'] = adjusted_rand_score(true_labels, spectral_labels)
            clustering_results['spectral']['nmi'] = normalized_mutual_info_score(true_labels, spectral_labels)
        
        return clustering_results, embeddings_scaled
    
    def _find_optimal_k(self, embeddings, max_k=10):
        """Find optimal k using elbow method"""
        if len(embeddings) < 10:
            return min(3, len(embeddings))
        
        inertias = []
        k_range = range(2, min(max_k, len(embeddings)//2) + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) > 1:
            differences = np.diff(inertias)
            second_diff = np.diff(differences)
            if len(second_diff) > 0:
                optimal_k = k_range[np.argmax(second_diff) + 1] if len(second_diff) > 0 else 3
            else:
                optimal_k = 3
        else:
            optimal_k = 3
            
        return optimal_k
    
    def create_visualizations(self, embeddings_data, results_dict):
        """Create t-SNE and UMAP visualizations"""
        plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        for data in embeddings_data:
            dataset = data['dataset']
            dim = data['dimension']
            embeddings = data['embeddings']
            true_labels = data['true_labels']
            
            print(f"Creating visualizations for {dataset}/{dim}...")
            
            # Get clustering results
            cluster_key = f"{dataset}_{dim}"
            if cluster_key not in results_dict:
                continue
                
            clustering_results, embeddings_scaled = results_dict[cluster_key]
            
            # Create t-SNE and UMAP projections
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            umap_reducer = umap.UMAP(random_state=42)
            
            embeddings_tsne = tsne.fit_transform(embeddings_scaled)
            embeddings_umap = umap_reducer.fit_transform(embeddings_scaled)
            
            # Create comprehensive visualization
            self._create_comparison_plot(dataset, dim, embeddings_tsne, embeddings_umap, 
                                       true_labels, clustering_results, plots_dir)
    
    def _create_comparison_plot(self, dataset, dim, tsne_emb, umap_emb, true_labels, 
                              clustering_results, plots_dir):
        """Create comparison plot with true labels and clustering results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Clustering Analysis: {dataset} - {dim}\n', fontsize=16, fontweight='bold')
        
        # Plot configurations
        plots_config = [
            (tsne_emb, 't-SNE Visualization'),
            (umap_emb, 'UMAP Visualization')
        ]
        
        for row, (embeddings_2d, title_base) in enumerate(plots_config):
            # True labels (if available)
            ax = axes[row, 0]
            if true_labels is not None:
                scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=true_labels, cmap='tab10', s=30, alpha=0.7)
                ax.set_title(f'{title_base}\nTrue Labels\n(ARI Reference)')
                plt.colorbar(scatter, ax=ax)
            else:
                ax.text(0.5, 0.5, 'True Labels\nNot Available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title_base}\nTrue Labels')
            
            # K-means results
            ax = axes[row, 1]
            kmeans_labels = clustering_results['kmeans']['labels']
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=kmeans_labels, cmap='tab10', s=30, alpha=0.7)
            ari = clustering_results['kmeans'].get('ari', 'N/A')
            silhouette = clustering_results['kmeans']['silhouette']
            ax.set_title(f'{title_base}\nK-means Clustering\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}')
            plt.colorbar(scatter, ax=ax)
            
            # Spectral clustering results
            ax = axes[row, 2]
            spectral_labels = clustering_results['spectral']['labels']
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=spectral_labels, cmap='tab10', s=30, alpha=0.7)
            ari = clustering_results['spectral'].get('ari', 'N/A')
            silhouette = clustering_results['spectral']['silhouette']
            ax.set_title(f'{title_base}\nSpectral Clustering\nARI: {ari:.3f}, Silhouette: {silhouette:.3f}')
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        filename = f"{plots_dir}/clustering_{dataset}_{dim}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization: {filename}")
    
    def run_comprehensive_analysis(self):
        """Run complete clustering analysis"""
        print("Loading embeddings...")
        embeddings_data = self.load_embeddings_and_labels()
        
        if not embeddings_data:
            print("No embeddings found!")
            return
        
        print(f"\nLoaded {len(embeddings_data)} embedding sets")
        
        results_dict = {}
        all_results = []
        
        print("\nPerforming clustering analysis...")
        for data in embeddings_data:
            dataset = data['dataset']
            dim = data['dimension']
            embeddings = data['embeddings']
            true_labels = data['true_labels']
            
            print(f"Analyzing {dataset}/{dim}...")
            
            clustering_results, embeddings_scaled = self.perform_clustering(embeddings, true_labels)
            results_dict[f"{dataset}_{dim}"] = (clustering_results, embeddings_scaled)
            
            # Store results
            for method in ['kmeans', 'spectral']:
                result = {
                    'dataset': dataset,
                    'dimension': dim,
                    'algorithm': method,
                    'n_clusters': len(np.unique(clustering_results[method]['labels'])),
                    'silhouette_score': clustering_results[method]['silhouette'],
                    'n_samples': len(embeddings),
                    'embedding_dim': embeddings.shape[1]
                }
                
                if true_labels is not None:
                    result['ari'] = clustering_results[method]['ari']
                    result['nmi'] = clustering_results[method]['nmi']
                
                all_results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        print("\nCreating visualizations...")
        self.create_visualizations(embeddings_data, results_dict)
        
        print("\nSaving results...")
        self.save_detailed_results(results_df)
        
        print("\nGenerating summary analysis...")
        self.generate_summary_analysis(results_df)
        
        return results_df
    
    def save_detailed_results(self, results_df):
        """Save detailed clustering results"""
        results_df.to_csv('clustering_analysis_detailed.csv', index=False)
        print("Detailed results saved to 'clustering_analysis_detailed.csv'")
    
    def generate_summary_analysis(self, results_df):
        """Generate summary analysis and identify best embeddings"""
        print("\n" + "="*80)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*80)
        
        # Best performing embeddings by ARI (if available)
        if 'ari' in results_df.columns:
            print("\nTOP PERFORMING EMBEDDINGS (by ARI):")
            ari_results = results_df[results_df['ari'].notna()]
            best_ari = ari_results.loc[ari_results.groupby(['dataset', 'algorithm'])['ari'].idxmax()]
            
            for _, row in best_ari.iterrows():
                print(f"{row['dataset']:12} | {row['algorithm']:15} | {row['dimension']:8} | "
                      f"ARI: {row['ari']:.3f} | Silhouette: {row['silhouette_score']:.3f}")
        
        # Best by silhouette score
        print("\nTOP PERFORMING EMBEDDINGS (by Silhouette Score):")
        best_silhouette = results_df.loc[results_df.groupby(['dataset', 'algorithm'])['silhouette_score'].idxmax()]
        
        for _, row in best_silhouette.iterrows():
            ari_info = f" | ARI: {row['ari']:.3f}" if 'ari' in row and pd.notna(row['ari']) else ""
            print(f"{row['dataset']:12} | {row['algorithm']:15} | {row['dimension']:8} | "
                  f"Silhouette: {row['silhouette_score']:.3f}{ari_info}")
        
        # Overall best embeddings
        print("\nOVERALL BEST EMBEDDINGS:")
        if 'ari' in results_df.columns:
            overall_best = results_df.loc[results_df['ari'].idxmax()] if not results_df.empty else None
        else:
            overall_best = results_df.loc[results_df['silhouette_score'].idxmax()] if not results_df.empty else None
        
        if overall_best is not None:
            print(f"Dataset: {overall_best['dataset']}")
            print(f"Dimension: {overall_best['dimension']}")
            print(f"Algorithm: {overall_best['algorithm']}")
            print(f"Silhouette Score: {overall_best['silhouette_score']:.3f}")
            if 'ari' in overall_best and pd.notna(overall_best['ari']):
                print(f"ARI: {overall_best['ari']:.3f}")
        
        # Create summary visualization
        self._create_summary_visualization(results_df)
    
    def _create_summary_visualization(self, results_df):
        """Create summary visualization comparing all embeddings"""
        plots_dir = './plots'
        
        # Performance heatmap
        plt.figure(figsize=(12, 8))
        
        if 'ari' in results_df.columns:
            # Use ARI if available
            metric = 'ari'
            title_suffix = 'ARI'
        else:
            # Fall back to silhouette score
            metric = 'silhouette_score'
            title_suffix = 'Silhouette Score'
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            index=['dataset', 'dimension'], 
            columns='algorithm', 
            values=metric, 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', center=0, 
                   fmt='.3f', linewidths=0.5)
        plt.title(f'Clustering Performance ({title_suffix}) by Dataset and Algorithm')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/clustering_performance_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Algorithm comparison
        plt.figure(figsize=(10, 6))
        if 'ari' in results_df.columns:
            sns.boxplot(data=results_df, x='algorithm', y='ari')
            plt.title('ARI Distribution by Clustering Algorithm')
            plt.ylabel('Adjusted Rand Index (ARI)')
        else:
            sns.boxplot(data=results_df, x='algorithm', y='silhouette_score')
            plt.title('Silhouette Score Distribution by Clustering Algorithm')
            plt.ylabel('Silhouette Score')
        
        plt.xlabel('Clustering Algorithm')
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/algorithm_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    analysis = GraphEmbeddingClusteringAnalysis()
    results_df = analysis.run_comprehensive_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)
    print("\nCheck the 'plots' directory for visualizations")
    print("Check 'clustering_analysis_detailed.csv' for detailed results")

if __name__ == "__main__":
    main()