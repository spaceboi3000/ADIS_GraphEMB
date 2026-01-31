import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (adjusted_rand_score,silhouette_score,normalized_mutual_info_score)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


@dataclass
class AnalysisConfig: # Clustering analysis Configuration
 
    base_path: str = '.'
    datasets: List[str] = field(default_factory=lambda: ['MUTAG', 'ENZYMES', 'IMDB-MULTI'])
    dimensions: List[str] = field(default_factory=lambda: ['dim64', 'dim128', 'dim256'])
    output_dir: str = './results'
    embedding_filename: str = 'Graph2Vec_embeddings'
    random_state: int = 42
    max_k_clusters: int = 10
    n_init_kmeans: int = 10
    
    def __post_init__(self):
        
        script_dir = Path(__file__).resolve().parent
        output_path = Path(self.output_dir)
        if not output_path.is_absolute():
            output_path = script_dir / output_path

        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'plots').mkdir(exist_ok=True)
        
        self.output_dir = str(output_path)


@dataclass
class EmbeddingData: 

    dataset: str
    dimension: str
    embeddings: np.ndarray
    true_labels: Optional[np.ndarray] = None
    file_type: str = 'npy'
    
    @property
    def n_samples(self) -> int: 
        return len(self.embeddings)
    
    @property
    def embedding_dim(self) -> int: 
        return self.embeddings.shape[1]
    
    @property
    def has_labels(self) -> bool: 
        return self.true_labels is not None


class LabelLoader:
    
    DATASET_LABELS = { 'MUTAG': lambda n: np.array([0] * 63 + [1] * 125)[:n],'ENZYMES': lambda n: np.array([i for i in range(6) for _ in range(100)])[:n],'IMDB-MULTI': lambda n: np.array([i for i in range(3) for _ in range(1000)])[:n] }
    
    @classmethod
    def load_labels(cls, dataset: str, n_samples: int) -> Optional[np.ndarray]:

        if dataset in cls.DATASET_LABELS:
            return cls.DATASET_LABELS[dataset](n_samples)
        return None
    
    @classmethod
    def register_dataset(cls, dataset: str, label_fn: callable):
        cls.DATASET_LABELS[dataset] = label_fn


class EmbeddingLoader:

    def __init__(self, config: AnalysisConfig):
        self.config = config
        
    def load_all(self) -> List[EmbeddingData]:

        embeddings_list = []
        for dataset in self.config.datasets:
            for dim in self.config.dimensions:
                embedding_data = self._load_single(dataset, dim)
                
                if embedding_data:
                    embeddings_list.append(embedding_data)
                    
                    print(f"Loaded {dataset}/{dim}: {embedding_data.embeddings.shape}, "
                          f"Labels: {embedding_data.has_labels}")
        
        return embeddings_list
    
    def _load_single(self, dataset: str, dimension: str) -> Optional[EmbeddingData]:
    
        base_dir = Path(self.config.base_path, dataset, dimension)
        
        npy_path = base_dir / f'{self.config.embedding_filename}.npy'
        if npy_path.exists():
            
            embeddings = np.load(npy_path)
            true_labels = LabelLoader.load_labels(dataset, len(embeddings))
            return EmbeddingData(dataset, dimension, embeddings, true_labels, 'npy')
        

        csv_path = base_dir / f'{self.config.embedding_filename}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            embeddings = self._extract_embeddings_from_csv(df)
            true_labels = LabelLoader.load_labels(dataset, len(embeddings))
            return EmbeddingData(dataset, dimension, embeddings, true_labels, 'csv')
        
        print(f"Warning?!?!?: No embedding file found for {dataset}/{dimension}")
        return None
    
    @staticmethod
    def _extract_embeddings_from_csv(df: pd.DataFrame) -> np.ndarray:
        id_columns = ['graph_id', 'GraphId', 'id', 'ID']
        for col in id_columns:
            if col in df.columns:
                
                return df.drop(columns=[col]).values
        
        return df.values


class ClusteringEngine:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        
        
    def cluster(self, embedding_data: EmbeddingData) -> Dict[str, Any]:
   
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embedding_data.embeddings)
        
        #number of clusters
        n_clusters = self._determine_n_clusters(embeddings_scaled, embedding_data)
        
        kmeans_labels = self._kmeans_clustering(embeddings_scaled, n_clusters)
        spectral_labels = self._spectral_clustering(embeddings_scaled, n_clusters)
        results = { 'embeddings_scaled': embeddings_scaled,'n_clusters': n_clusters,'kmeans': self._calculate_metrics(  embeddings_scaled, kmeans_labels, embedding_data.true_labels ), 'spectral': self._calculate_metrics(embeddings_scaled, spectral_labels, embedding_data.true_labels) }
        
        return results
    
    def _determine_n_clusters(self, embeddings: np.ndarray,  embedding_data: EmbeddingData) -> int:
      
        if embedding_data.has_labels:
            return len(np.unique(embedding_data.true_labels))
        return self._find_optimal_k_elbow(embeddings)
    
    def _find_optimal_k_elbow(self, embeddings: np.ndarray) -> int: #Find optimal k using elbow method. returnf oprimal number of cluster
    
        n_samples = len(embeddings)
        if n_samples < 10:
            return min(3, n_samples)
        
        max_k = min(self.config.max_k_clusters, n_samples // 2)
        k_range = range(2, max_k + 1)
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state,n_init=self.config.n_init_kmeans)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)
        
        # Detect elbow using second derivative
        if len(inertias) > 2:
            second_diff = np.diff(np.diff(inertias))
            optimal_idx = np.argmax(second_diff) + 1
            return list(k_range)[optimal_idx]
        
        return 3
    
    def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
       
        kmeans = KMeans(n_clusters=n_clusters,  random_state=self.config.random_state,n_init=self.config.n_init_kmeans)
        return kmeans.fit_predict(embeddings)
    
    def _spectral_clustering(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=self.config.random_state,     affinity='rbf', gamma=1.0)
        return spectral.fit_predict(embeddings)
    
    
    def _calculate_metrics(self, embeddings: np.ndarray, labels: np.ndarray, true_labels: Optional[np.ndarray]) -> Dict[str, Any]:
   
        metrics = {
            'labels': labels,
            'silhouette': silhouette_score(embeddings, labels)
        }
        
        if true_labels is not None:
            metrics['ari'] = adjusted_rand_score(true_labels, labels)
            metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
        
        return metrics


class Visualizer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.plots_dir = Path(config.output_dir, 'plots')
     
        
    def create_all_visualizations(self, embedding_data: EmbeddingData,  clustering_results: Dict[str, Any]):
     
        embeddings_scaled = clustering_results['embeddings_scaled']
        
        tsne_emb = self._compute_tsne(embeddings_scaled)
        umap_emb = self._compute_umap(embeddings_scaled)
        self._create_comparison_plot( embedding_data, clustering_results, tsne_emb, umap_emb)
    
    def _compute_tsne(self, embeddings: np.ndarray) -> np.ndarray:
      
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=self.config.random_state,perplexity=perplexity)
        return tsne.fit_transform(embeddings)
    
    
    
    def _compute_umap(self, embeddings: np.ndarray) -> np.ndarray:
   
        reducer = umap.UMAP(random_state=self.config.random_state)
        return reducer.fit_transform(embeddings)
    
    
    def _create_comparison_plot(self, embedding_data: EmbeddingData, clustering_results: Dict[str, Any], tsne_emb: np.ndarray, umap_emb: np.ndarray):
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Clustering Analysis: {embedding_data.dataset} - {embedding_data.dimension}', fontsize=16, fontweight='bold')
        projections = [ (tsne_emb, 't-SNE'),(umap_emb, 'UMAP') ]
        
        for row, (projection, method) in enumerate(projections):
            self._plot_row(axes[row], projection, method, embedding_data, clustering_results)
    
        plt.tight_layout()
        filename = self.plots_dir / f"clustering_{embedding_data.dataset}_{embedding_data.dimension}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Saved: {filename.name}")
    
    
    
    
    def _plot_row(self, axes_row, projection: np.ndarray, method: str,embedding_data: EmbeddingData, clustering_results: Dict[str, Any]):
        
        # True labels
        self._plot_true_labels(axes_row[0], projection, method, embedding_data)
        # K-means
        self._plot_clustering(axes_row[1], projection, method, 'K-means',clustering_results['kmeans'])
        # Spectral
        self._plot_clustering(axes_row[2], projection, method, 'Spectral', clustering_results['spectral'])
    
    
    def _plot_true_labels(self, ax, projection: np.ndarray, method: str,   embedding_data: EmbeddingData):
        if embedding_data.has_labels:
            
            scatter = ax.scatter(projection[:, 0], projection[:, 1], c=embedding_data.true_labels, cmap='tab10', s=30, alpha=0.7)
            ax.set_title(f'{method}\nTrue Labels')
            plt.colorbar(scatter, ax=ax)
        else:
            ax.text(0.5, 0.5, 'True Labels\nNot Available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method}\nTrue Labels')
    
    
    
    def _plot_clustering(self, ax, projection: np.ndarray, method: str, algorithm: str, results: Dict[str, Any]):
        
        scatter = ax.scatter(projection[:, 0], projection[:, 1], c=results['labels'], cmap='tab10', s=30, alpha=0.7)
        ari = f"{results['ari']:.3f}" if 'ari' in results else 'N/A'
        sil = f"{results['silhouette']:.3f}"
        ax.set_title(f'{method}\n{algorithm} Clustering\nARI: {ari} | Sil: {sil}')
        plt.colorbar(scatter, ax=ax)
    
    
    def create_summary_visualizations(self, results_df: pd.DataFrame):
        self._create_performance_heatmap(results_df)
        self._create_algorithm_comparison(results_df)
    
    
    
    
    def _create_performance_heatmap(self, results_df: pd.DataFrame):
  
        plt.figure(figsize=(12, 8))
        
        metric = 'ari' if 'ari' in results_df.columns else 'silhouette_score'
        title_suffix = 'ARI' if metric == 'ari' else 'Silhouette Score'
        pivot_data = results_df.pivot_table( index=['dataset', 'dimension'],columns='algorithm',values=metric, aggfunc='mean')
        
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd',fmt='.3f', linewidths=0.5)
        plt.title(f'Clustering Performance ({title_suffix})')
        plt.tight_layout()
        
        filename = self.plots_dir / 'performance_heatmap.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved: {filename.name}")
    
    
    
    
    def _create_algorithm_comparison(self, results_df: pd.DataFrame):
        """Create algorithm comparison boxplot."""
        plt.figure(figsize=(10, 6))
        
        if 'ari' in results_df.columns:
            sns.boxplot(data=results_df, x='algorithm', y='ari')
            plt.ylabel('Adjusted Rand Index (ARI)')
            plt.title('ARI Distribution by Algorithm')
        else:
            sns.boxplot(data=results_df, x='algorithm', y='silhouette_score')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score Distribution by Algorithm')
        
        plt.xlabel('Clustering Algorithm')
        plt.tight_layout()
        
        filename = self.plots_dir / 'algorithm_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename.name}")




class ResultsManager:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results_dir = Path(config.output_dir)
        
    def compile_results(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(all_results)
    
    def save_results(self, results_df: pd.DataFrame):
      
        filename = self.results_dir / 'clustering_analysis_detailed.csv'
        results_df.to_csv(filename, index=False)
        print(f"Results saved: {filename}")
    
    def generate_summary_report(self, results_df: pd.DataFrame):
       
        print("\n" + "=" * 80)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("=" * 80)
        
        self._print_best_by_ari(results_df)
        self._print_best_by_silhouette(results_df)
        self._print_overall_best(results_df)
    
    def _print_best_by_ari(self, results_df: pd.DataFrame):
 
        if 'ari' not in results_df.columns:
            return
        
        print("\nTOP PERFORMING EMBEDDINGS (by ARI):")
        print("-" * 80)
        
        ari_results = results_df[results_df['ari'].notna()]
        best_ari = ari_results.loc[
            ari_results.groupby(['dataset', 'algorithm'])['ari'].idxmax()
        ]
        
        for _, row in best_ari.iterrows():
            print(f"{row['dataset']:12} | {row['algorithm']:10} | "
                  f"{row['dimension']:8} | ARI: {row['ari']:.3f} | "
                  f"Sil: {row['silhouette_score']:.3f}")
    
    def _print_best_by_silhouette(self, results_df: pd.DataFrame):

        print("\nTOP PERFORMING EMBEDDINGS (by Silhouette Score):")
        print("-" * 80)
        
        best_silhouette = results_df.loc[
            results_df.groupby(['dataset', 'algorithm'])['silhouette_score'].idxmax()
        ]
        
        for _, row in best_silhouette.iterrows():
            ari_info = f" | ARI: {row['ari']:.3f}" if 'ari' in row and pd.notna(row['ari']) else ""
            print(f"{row['dataset']:12} | {row['algorithm']:10} | "
                  f"{row['dimension']:8} | Sil: {row['silhouette_score']:.3f}{ari_info}")
    
    def _print_overall_best(self, results_df: pd.DataFrame):
        print("\nOVERALL BEST EMBEDDING:")
        print("-" * 80)
        
        if 'ari' in results_df.columns and not results_df['ari'].isna().all():
            best = results_df.loc[results_df['ari'].idxmax()]
            metric_name = 'ARI'
            metric_value = best['ari']
        else:
            best = results_df.loc[results_df['silhouette_score'].idxmax()]
            metric_name = 'Silhouette Score'
            metric_value = best['silhouette_score']
        
        print(f"Dataset: {best['dataset']}")
        print(f"Dimension: {best['dimension']}")
        print(f"Algorithm: {best['algorithm']}")
        print(f"{metric_name}: {metric_value:.3f}")
        print(f"Silhouette Score: {best['silhouette_score']:.3f}")
        if 'ari' in best and pd.notna(best['ari']):
            print(f"NMI: {best.get('nmi', 'N/A'):.3f}")


class ClusteringAnalyzer: #Main analysis pipeline

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.loader = EmbeddingLoader(config)
        self.clustering_engine = ClusteringEngine(config)
        self.visualizer = Visualizer(config)
        self.results_manager = ResultsManager(config)
        
    def run_analysis(self) -> pd.DataFrame:
        print("=" * 80)
        print("GRAPH EMBEDDING CLUSTERING ANALYSIS")
        print("=" * 80)
        print("\n[1/4] Loading embeddings...")
        embeddings_list = self.loader.load_all()
        
        if not embeddings_list:
            print("No embeddings found..")
            return pd.DataFrame()
        
        print(f" Loaded {len(embeddings_list)} embedding sets")
        print("\n[2/4] Performing clustering analysis...")
        all_results = []
        
        for embedding_data in embeddings_list:
           
            print(f" Analyzing {embedding_data.dataset}/{embedding_data.dimension}")
            clustering_results = self.clustering_engine.cluster(embedding_data)
            
            #storing results
            for algorithm in ['kmeans', 'spectral']:
                result = { 'dataset': embedding_data.dataset, 'dimension': embedding_data.dimension,'algorithm': algorithm,'n_clusters': clustering_results['n_clusters'], 'silhouette_score': clustering_results[algorithm]['silhouette'],'n_samples': embedding_data.n_samples,'embedding_dim': embedding_data.embedding_dim }
                
                if 'ari' in clustering_results[algorithm]:
                    result['ari'] = clustering_results[algorithm]['ari']
                    result['nmi'] = clustering_results[algorithm]['nmi']
                
                all_results.append(result)
            
            # visualizations
            self.visualizer.create_all_visualizations(embedding_data, clustering_results)
        
        #compile results
        results_df = self.results_manager.compile_results(all_results)
        
        print("\n[3/4] Saving results...")
        self.results_manager.save_results(results_df)
        
        print("\n[4/4] Generating summary visualizations...")
        self.visualizer.create_summary_visualizations(results_df)
        self.results_manager.generate_summary_report(results_df)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED!")
        print("=" * 80)
        print(f"\nResults directory: {self.config.output_dir}")
        print(f"  - Detailed CSV: clustering_analysis_detailed.csv")
        print(f"  - Visualizations: plots/")
        
        return results_df


def main():

    config = AnalysisConfig( base_path='../embeddings/embeddings_graph2vec', datasets=['MUTAG', 'ENZYMES', 'IMDB-MULTI'],dimensions=['dim64', 'dim128', 'dim256'],output_dir='../csvs/clustering_results', random_state=42)
    
    analyzer = ClusteringAnalyzer(config)   #rrun analysis
    results = analyzer.run_analysis()
    
    return results


if __name__ == "__main__":
    main()