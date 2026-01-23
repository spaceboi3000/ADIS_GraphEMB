"""
Extended Clustering Visualizations with t-SNE and UMAP
For academic paper figures - compact, information-dense layouts

Figure Types:
=============
SINGLE-ROW FIGURES (for individual chapters - original OR permuted):
- Method comparison: 1×3 grid (GIN, Graph2Vec, NetLSD)
- Dataset comparison: 1×3 grid (ENZYMES, IMDB-MULTI, MUTAG)
- Dimension comparison: 1×3 grid (dim64, dim128, dim256)

COMPARISON FIGURES (for comparison chapter - original VS permuted):
- Method comparison: 2×3 grid (Original/Permuted × Methods)
- Comprehensive grid: 3×6 grid (Methods × Datasets×Orig/Perm)

COMPREHENSIVE FIGURES:
- Methods × Dimensions: 3×3 grid for single dataset
- t-SNE vs UMAP: side-by-side comparison across all methods/datasets

Usage:
    python clustering_vis_extended.py
    python clustering_vis_extended.py --embeddings-dir ../embeddings --permuted-dir ../permutated_embeddings

Requirements:
    pip install numpy pandas matplotlib seaborn scikit-learn umap-learn
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings('ignore')

# Try importing UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration for visualizations."""
    PROJECT_ROOT = Path(__file__).parent.parent
    EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
    PERMUTED_EMBEDDINGS_DIR = PROJECT_ROOT / "permutated_embeddings"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    DATASETS_DIR = PROJECT_ROOT / "DATASETS"
    METRICS_CSV = SCRIPTS_DIR / "clustering_merged.csv"
    
    # Visualization settings
    RANDOM_STATE = 42
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    
    # Academic paper plot sizes (in inches)
    FIGSIZE_1x3 = (7, 2.3)         # Single row, 3 columns
    FIGSIZE_2x3 = (7, 4.5)         # Two rows, 3 columns
    FIGSIZE_3x3 = (7, 6.5)         # Three rows, 3 columns
    FIGSIZE_3x6 = (10, 6.5)        # Three rows, 6 columns (comprehensive)
    FIGSIZE_2x2 = (5, 4.5)         # 2x2 grid
    
    DPI = 300
    FONT_SIZE = 8
    TITLE_SIZE = 9
    LABEL_SIZE = 7
    LEGEND_SIZE = 6
    TICK_SIZE = 6
    MARKER_SIZE = 8
    ALPHA = 0.6
    
    # Color schemes
    DATASET_COLORS = {
        'ENZYMES': '#1f77b4',
        'IMDB-MULTI': '#2ca02c',
        'MUTAG': '#d62728'
    }
    
    METHOD_COLORS = {
        'GIN': '#1f77b4',
        'Graph2Vec': '#ff7f0e',
        'NetLSD': '#2ca02c'
    }
    
    CLUSTER_PALETTE = sns.color_palette("colorblind", 10)
    
    DATASETS = ['ENZYMES', 'IMDB-MULTI', 'MUTAG']
    METHODS = ['GIN', 'Graph2Vec', 'NetLSD']
    DIMS = [64, 128, 256]


def setup_matplotlib_style():
    """Set up matplotlib for academic paper figures."""
    plt.rcParams.update({
        'font.size': Config.FONT_SIZE,
        'axes.titlesize': Config.TITLE_SIZE,
        'axes.labelsize': Config.LABEL_SIZE,
        'xtick.labelsize': Config.TICK_SIZE,
        'ytick.labelsize': Config.TICK_SIZE,
        'legend.fontsize': Config.LEGEND_SIZE,
        'figure.dpi': Config.DPI,
        'savefig.dpi': Config.DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'sans-serif',
    })


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_metrics(csv_path=None):
    """Load clustering metrics from CSV."""
    if csv_path is None:
        csv_path = Config.METRICS_CSV
    return pd.read_csv(csv_path)


def get_best_config_for_method_dataset_dim(df, method, dataset, dim, cluster_alg='kmeans', metric='ari'):
    """Get the run configuration with best ARI for a given method/dataset/dim."""
    mask = (df['method'] == method) & (df['dataset'] == dataset) & (df['cluster_alg'] == cluster_alg)
    
    if method == 'GIN':
        if dataset == 'MUTAG':
            dim_map = {64: 323, 128: 643, 256: 1283}
        elif dataset == 'IMDB-MULTI':
            dim_map = {64: 192, 128: 384, 256: 768}
        else:  # ENZYMES
            dim_map = {64: 384, 128: 768, 256: 1536}
        actual_dim = dim_map.get(dim, dim)
        subset = df[mask & (df['dim'] == actual_dim)]
    else:
        subset = df[mask & (df['dim'] == dim)]
    
    if subset.empty:
        return None
    
    best_idx = subset[metric].idxmax()
    return subset.loc[best_idx]


def load_labels_for_dataset(dataset, datasets_dir):
    """Load ground truth labels for a dataset."""
    labels_file = datasets_dir / dataset / "labels.csv"
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file)
        if 'label' in labels_df.columns:
            return labels_df['label'].values
        return labels_df.iloc[:, 0].values
    return None


def load_gin_embeddings(dataset, run_name, base_dir, datasets_dir):
    """Load GIN embeddings combining train/val/test splits."""
    run_dir = base_dir / "embeddings_gin" / dataset / run_name
    
    embeddings_list = []
    labels_list = []
    
    for split in ['train', 'val', 'test']:
        emb_file = run_dir / f"{split}_embeddings.npy"
        labels_file = run_dir / f"{split}_labels.csv"
        
        if emb_file.exists() and labels_file.exists():
            emb = np.load(emb_file)
            labels = pd.read_csv(labels_file)
            embeddings_list.append(emb)
            if 'label' in labels.columns:
                labels_list.append(labels['label'].values)
            else:
                labels_list.append(labels.iloc[:, 0].values)
    
    if not embeddings_list:
        return None, None
    
    return np.vstack(embeddings_list), np.concatenate(labels_list)


def load_graph2vec_embeddings(dataset, dim, base_dir, datasets_dir):
    """Load Graph2Vec embeddings."""
    emb_file = base_dir / "embeddings_graph2vec" / dataset / f"dim{dim}" / "Graph2Vec_embeddings.npy"
    
    if not emb_file.exists():
        return None, None
    
    embeddings = np.load(emb_file)
    labels = load_labels_for_dataset(dataset, datasets_dir)
    
    if labels is None:
        labels = np.zeros(len(embeddings))
    
    return embeddings, labels


def load_netlsd_embeddings(dataset, dim, base_dir, datasets_dir):
    """Load NetLSD embeddings."""
    emb_file = base_dir / "embeddings_netlsd" / dataset / f"dim{dim}" / "NetLSD_embeddings.npy"
    
    if not emb_file.exists():
        return None, None
    
    embeddings = np.load(emb_file)
    labels = load_labels_for_dataset(dataset, datasets_dir)
    
    if labels is None:
        labels = np.zeros(len(embeddings))
    
    return embeddings, labels


def load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name=None):
    """Generic embedding loader."""
    if method == 'GIN':
        if run_name is None:
            return None, None
        return load_gin_embeddings(dataset, run_name, base_dir, datasets_dir)
    elif method == 'Graph2Vec':
        return load_graph2vec_embeddings(dataset, dim, base_dir, datasets_dir)
    elif method == 'NetLSD':
        return load_netlsd_embeddings(dataset, dim, base_dir, datasets_dir)
    return None, None


# =============================================================================
# Dimensionality Reduction
# =============================================================================

def compute_tsne(embeddings, perplexity=None, random_state=None):
    """Compute t-SNE projection."""
    if perplexity is None:
        perplexity = min(Config.TSNE_PERPLEXITY, len(embeddings) - 1)
    if random_state is None:
        random_state = Config.RANDOM_STATE
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Handle different scikit-learn versions
    import sklearn
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    
    if sklearn_version >= (1, 5):
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=Config.TSNE_N_ITER,
                    random_state=random_state, init='pca')
    else:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=Config.TSNE_N_ITER,
                    random_state=random_state, init='pca')
    
    return tsne.fit_transform(embeddings_scaled)


def compute_umap(embeddings, n_neighbors=None, min_dist=None, random_state=None):
    """Compute UMAP projection."""
    if not UMAP_AVAILABLE:
        return None
    
    if n_neighbors is None:
        n_neighbors = min(Config.UMAP_N_NEIGHBORS, len(embeddings) - 1)
    if min_dist is None:
        min_dist = Config.UMAP_MIN_DIST
    if random_state is None:
        random_state = Config.RANDOM_STATE
    
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    
    return reducer.fit_transform(embeddings_scaled)


def compute_projection(embeddings, projection_type='tsne'):
    """Compute projection with fallback."""
    if projection_type == 'umap':
        proj = compute_umap(embeddings)
        if proj is not None:
            return proj
    return compute_tsne(embeddings)


# =============================================================================
# Plotting Helpers
# =============================================================================

def plot_embedding_scatter(ax, projection, labels, title=None, metrics_text=None,
                          show_legend=False, palette=None, marker_size=None):
    """Plot a single embedding scatter plot."""
    if palette is None:
        palette = Config.CLUSTER_PALETTE
    if marker_size is None:
        marker_size = Config.MARKER_SIZE
    
    unique_labels = np.unique(labels)
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(projection[mask, 0], projection[mask, 1],
                   c=[palette[i % len(palette)]], label=f'Class {int(label)}',
                   s=marker_size, alpha=Config.ALPHA, edgecolors='none')
    
    if title:
        ax.set_title(title, fontsize=Config.TITLE_SIZE, fontweight='bold')
    
    if metrics_text:
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                fontsize=Config.LEGEND_SIZE, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if show_legend:
        ax.legend(loc='lower right', framealpha=0.8, edgecolor='none',
                 markerscale=1.5, handletextpad=0.1)


def plot_empty_cell(ax, message='N/A', title=None):
    """Plot an empty cell with message."""
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=Config.TITLE_SIZE, fontweight='bold')


# =============================================================================
# SINGLE-ROW FIGURES (for individual chapters)
# =============================================================================

def create_method_comparison_single(df, dataset, dim, base_dir, datasets_dir,
                                    projection_type='tsne', output_path=None):
    """
    Compare all methods for a single dataset - SINGLE ROW.
    Layout: 1×3 (GIN, Graph2Vec, NetLSD)
    """
    fig, axes = plt.subplots(1, 3, figsize=Config.FIGSIZE_1x3)
    
    for col, method in enumerate(Config.METHODS):
        ax = axes[col]
        best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
        
        if best_config is None:
            plot_empty_cell(ax, 'No data', method)
            continue
        
        run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
        embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
        
        if embeddings is None:
            plot_empty_cell(ax, 'Missing', method)
            continue
        
        projection = compute_projection(embeddings, projection_type)
        metrics_text = f"ARI={best_config['ari']:.3f}\nNMI={best_config['nmi']:.3f}"
        
        plot_embedding_scatter(ax, projection, labels, title=method,
                              metrics_text=metrics_text, show_legend=(col == 2))
    
    fig.suptitle(f"{dataset} (dim={dim}, {projection_type.upper()})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_dataset_comparison_single(df, method, dim, base_dir, datasets_dir,
                                     projection_type='tsne', output_path=None):
    """
    Compare all datasets for a single method - SINGLE ROW.
    Layout: 1×3 (ENZYMES, IMDB-MULTI, MUTAG)
    """
    fig, axes = plt.subplots(1, 3, figsize=Config.FIGSIZE_1x3)
    
    for col, dataset in enumerate(Config.DATASETS):
        ax = axes[col]
        best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
        
        if best_config is None:
            plot_empty_cell(ax, 'No data', dataset)
            continue
        
        run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
        embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
        
        if embeddings is None:
            plot_empty_cell(ax, 'Missing', dataset)
            continue
        
        projection = compute_projection(embeddings, projection_type)
        metrics_text = f"ARI={best_config['ari']:.3f}\nNMI={best_config['nmi']:.3f}"
        
        plot_embedding_scatter(ax, projection, labels, title=dataset,
                              metrics_text=metrics_text, show_legend=(col == 2))
    
    fig.suptitle(f"{method} (dim={dim}, {projection_type.upper()})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_dimension_comparison_single(df, method, dataset, base_dir, datasets_dir,
                                       projection_type='tsne', output_path=None):
    """
    Compare all dimensions for a single method/dataset - SINGLE ROW.
    Layout: 1×3 (dim64, dim128, dim256)
    """
    fig, axes = plt.subplots(1, 3, figsize=Config.FIGSIZE_1x3)
    
    for col, dim in enumerate(Config.DIMS):
        ax = axes[col]
        best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
        
        if best_config is None:
            plot_empty_cell(ax, 'No data', f'dim={dim}')
            continue
        
        run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
        embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
        
        if embeddings is None:
            plot_empty_cell(ax, 'Missing', f'dim={dim}')
            continue
        
        projection = compute_projection(embeddings, projection_type)
        metrics_text = f"ARI={best_config['ari']:.3f}\nNMI={best_config['nmi']:.3f}"
        
        plot_embedding_scatter(ax, projection, labels, title=f'dim={dim}',
                              metrics_text=metrics_text, show_legend=(col == 2))
    
    fig.suptitle(f"{method} on {dataset} ({projection_type.upper()})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# COMPARISON FIGURES (Original vs Permuted)
# =============================================================================

def create_method_comparison_dual(df, dataset, dim, base_dir, datasets_dir,
                                  permuted_base_dir, projection_type='tsne', output_path=None):
    """
    Compare methods: Original vs Permuted.
    Layout: 2×3 (Original/Permuted × GIN/Graph2Vec/NetLSD)
    """
    fig, axes = plt.subplots(2, 3, figsize=Config.FIGSIZE_2x3)
    
    dirs = [('Original', base_dir), ('Permuted', permuted_base_dir)]
    
    for col, method in enumerate(Config.METHODS):
        for row, (label, b_dir) in enumerate(dirs):
            ax = axes[row, col]
            
            if b_dir is None:
                plot_empty_cell(ax, 'N/A', method if row == 0 else None)
                if col == 0:
                    ax.set_ylabel(label, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
            
            if best_config is None:
                plot_empty_cell(ax, 'No data', method if row == 0 else None)
                if col == 0:
                    ax.set_ylabel(label, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
            embeddings, labels = load_embeddings(method, dataset, dim, b_dir, datasets_dir, run_name)
            
            if embeddings is None:
                plot_empty_cell(ax, 'Missing', method if row == 0 else None)
                if col == 0:
                    ax.set_ylabel(label, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            projection = compute_projection(embeddings, projection_type)
            metrics_text = f"ARI={best_config['ari']:.3f}"
            
            title = method if row == 0 else None
            plot_embedding_scatter(ax, projection, labels, title=title,
                                  metrics_text=metrics_text, show_legend=(row == 0 and col == 2))
            
            if col == 0:
                ax.set_ylabel(label, fontsize=Config.LABEL_SIZE, fontweight='bold')
    
    fig.suptitle(f"{dataset} - dim={dim} ({projection_type.upper()})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_comprehensive_comparison_grid(df, base_dir, datasets_dir, permuted_base_dir,
                                         projection_type='tsne', dim=128, output_path=None):
    """
    Comprehensive comparison: Methods × (Datasets × Orig/Perm).
    Layout: 3×6 grid
    """
    fig = plt.figure(figsize=Config.FIGSIZE_3x6)
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.08, hspace=0.25)
    
    for m_idx, method in enumerate(Config.METHODS):
        for d_idx, dataset in enumerate(Config.DATASETS):
            for p_idx, (label, b_dir) in enumerate([('O', base_dir), ('P', permuted_base_dir)]):
                col = d_idx * 2 + p_idx
                ax = fig.add_subplot(gs[m_idx, col])
                
                if b_dir is None:
                    plot_empty_cell(ax, 'N/A')
                    if m_idx == 0:
                        ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(method, fontsize=8, fontweight='bold')
                    continue
                
                best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
                
                if best_config is None:
                    plot_empty_cell(ax, 'N/A')
                    if m_idx == 0:
                        ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(method, fontsize=8, fontweight='bold')
                    continue
                
                run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
                embeddings, labels = load_embeddings(method, dataset, dim, b_dir, datasets_dir, run_name)
                
                if embeddings is None:
                    plot_empty_cell(ax, 'Missing')
                    if m_idx == 0:
                        ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(method, fontsize=8, fontweight='bold')
                    continue
                
                projection = compute_projection(embeddings, projection_type)
                metrics_text = f"ARI={best_config['ari']:.2f}"
                
                plot_embedding_scatter(ax, projection, labels, metrics_text=metrics_text, marker_size=5)
                
                if m_idx == 0:
                    ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(method, fontsize=8, fontweight='bold')
    
    fig.suptitle(f"Embedding Comparison: Original (O) vs Permuted (P) - dim={dim}, {projection_type.upper()}",
                fontsize=10, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# METHODS × DIMENSIONS GRID (3×3)
# =============================================================================

def create_methods_dimensions_grid(df, dataset, base_dir, datasets_dir,
                                   projection_type='tsne', output_path=None):
    """
    Methods × Dimensions grid for a single dataset.
    Layout: 3×3 (Methods × Dimensions)
    """
    fig, axes = plt.subplots(3, 3, figsize=Config.FIGSIZE_3x3)
    
    for m_idx, method in enumerate(Config.METHODS):
        for d_idx, dim in enumerate(Config.DIMS):
            ax = axes[m_idx, d_idx]
            
            best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
            
            if best_config is None:
                plot_empty_cell(ax, 'No data')
                if m_idx == 0:
                    ax.set_title(f'dim={dim}', fontsize=Config.TITLE_SIZE, fontweight='bold')
                if d_idx == 0:
                    ax.set_ylabel(method, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
            embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
            
            if embeddings is None:
                plot_empty_cell(ax, 'Missing')
                if m_idx == 0:
                    ax.set_title(f'dim={dim}', fontsize=Config.TITLE_SIZE, fontweight='bold')
                if d_idx == 0:
                    ax.set_ylabel(method, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            projection = compute_projection(embeddings, projection_type)
            metrics_text = f"ARI={best_config['ari']:.3f}"
            
            plot_embedding_scatter(ax, projection, labels, metrics_text=metrics_text,
                                  show_legend=(m_idx == 0 and d_idx == 2))
            
            if m_idx == 0:
                ax.set_title(f'dim={dim}', fontsize=Config.TITLE_SIZE, fontweight='bold')
            if d_idx == 0:
                ax.set_ylabel(method, fontsize=Config.LABEL_SIZE, fontweight='bold')
    
    fig.suptitle(f"{dataset} - Methods × Dimensions ({projection_type.upper()})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# t-SNE vs UMAP COMPREHENSIVE COMPARISON
# =============================================================================

def create_tsne_umap_comparison_grid(df, base_dir, datasets_dir, dim=128, output_path=None):
    """
    Comprehensive t-SNE vs UMAP comparison across all methods and datasets.
    Layout: 3×6 (Methods × Datasets×TSNE/UMAP)
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping t-SNE vs UMAP comparison.")
        return None
    
    fig = plt.figure(figsize=Config.FIGSIZE_3x6)
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.08, hspace=0.25)
    
    for m_idx, method in enumerate(Config.METHODS):
        for d_idx, dataset in enumerate(Config.DATASETS):
            best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
            
            run_name = None
            embeddings, labels = None, None
            
            if best_config is not None:
                run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
                embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
            
            for p_idx, proj_type in enumerate(['tsne', 'umap']):
                col = d_idx * 2 + p_idx
                ax = fig.add_subplot(gs[m_idx, col])
                
                if embeddings is None:
                    plot_empty_cell(ax, 'N/A')
                    if m_idx == 0:
                        label = 't-SNE' if proj_type == 'tsne' else 'UMAP'
                        ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(method, fontsize=8, fontweight='bold')
                    continue
                
                projection = compute_projection(embeddings, proj_type)
                metrics_text = f"ARI={best_config['ari']:.2f}"
                
                plot_embedding_scatter(ax, projection, labels, metrics_text=metrics_text, marker_size=5)
                
                if m_idx == 0:
                    label = 't-SNE' if proj_type == 'tsne' else 'UMAP'
                    ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(method, fontsize=8, fontweight='bold')
    
    fig.suptitle(f"t-SNE vs UMAP Comparison (dim={dim})",
                fontsize=10, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_tsne_umap_single_method(df, method, base_dir, datasets_dir, dim=128, output_path=None):
    """
    t-SNE vs UMAP for a single method across all datasets.
    Layout: 2×3 (t-SNE/UMAP × Datasets)
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available.")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=Config.FIGSIZE_2x3)
    
    for d_idx, dataset in enumerate(Config.DATASETS):
        best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
        
        run_name = None
        embeddings, labels = None, None
        
        if best_config is not None:
            run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
            embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
        
        for p_idx, proj_type in enumerate(['tsne', 'umap']):
            ax = axes[p_idx, d_idx]
            
            if embeddings is None:
                plot_empty_cell(ax, 'N/A', dataset if p_idx == 0 else None)
                if d_idx == 0:
                    ax.set_ylabel('t-SNE' if proj_type == 'tsne' else 'UMAP',
                                 fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            projection = compute_projection(embeddings, proj_type)
            metrics_text = f"ARI={best_config['ari']:.3f}"
            
            title = dataset if p_idx == 0 else None
            plot_embedding_scatter(ax, projection, labels, title=title,
                                  metrics_text=metrics_text, show_legend=(p_idx == 0 and d_idx == 2))
            
            if d_idx == 0:
                ax.set_ylabel('t-SNE' if proj_type == 'tsne' else 'UMAP',
                             fontsize=Config.LABEL_SIZE, fontweight='bold')
    
    fig.suptitle(f"{method} - t-SNE vs UMAP (dim={dim})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# METRICS HEATMAP
# =============================================================================

def create_metrics_heatmap(df, cluster_alg='kmeans', output_path=None):
    """Create heatmap showing ARI/NMI across methods and datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    
    for ax, metric, title in zip(axes, ['ari', 'nmi'],
                                  ['ARI (Adjusted Rand Index)', 'NMI (Normalized Mutual Info)']):
        pivot = df[df['cluster_alg'] == cluster_alg].groupby(['method', 'dataset'])[metric].max().unstack()
        
        methods_order = [m for m in Config.METHODS if m in pivot.index]
        datasets_order = [d for d in Config.DATASETS if d in pivot.columns]
        pivot = pivot.loc[methods_order, datasets_order]
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                   vmin=0, vmax=max(0.5, pivot.max().max()),
                   cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
        
        ax.set_title(title, fontsize=Config.TITLE_SIZE, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    fig.suptitle(f"Clustering Performance ({cluster_alg})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# NEW: t-SNE vs UMAP PER DATASET (2×3)
# =============================================================================

def create_tsne_vs_umap_per_dataset(df, dataset, dim, base_dir, datasets_dir, output_path=None):
    """
    t-SNE vs UMAP comparison for a single dataset.
    Layout: 2×3 (rows: t-SNE/UMAP, cols: GIN/Graph2Vec/NetLSD)
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available.")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=Config.FIGSIZE_2x3)
    
    for col, method in enumerate(Config.METHODS):
        best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
        
        embeddings, labels = None, None
        if best_config is not None:
            run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
            embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
        
        for row, proj_type in enumerate(['tsne', 'umap']):
            ax = axes[row, col]
            
            if embeddings is None:
                plot_empty_cell(ax, 'N/A', method if row == 0 else None)
                if col == 0:
                    ax.set_ylabel('t-SNE' if proj_type == 'tsne' else 'UMAP',
                                 fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            projection = compute_projection(embeddings, proj_type)
            metrics_text = f"ARI={best_config['ari']:.3f}\nNMI={best_config['nmi']:.3f}"
            
            title = method if row == 0 else None
            plot_embedding_scatter(ax, projection, labels, title=title,
                                  metrics_text=metrics_text, show_legend=(row == 0 and col == 2))
            
            if col == 0:
                ax.set_ylabel('t-SNE' if proj_type == 'tsne' else 'UMAP',
                             fontsize=Config.LABEL_SIZE, fontweight='bold')
    
    fig.suptitle(f"{dataset} - t-SNE vs UMAP (dim={dim})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# NEW: K-MEANS vs SPECTRAL PER DATASET (2×3)
# =============================================================================

def create_kmeans_vs_spectral_per_dataset(df, dataset, dim, base_dir, datasets_dir,
                                          projection_type='tsne', output_path=None):
    """
    K-means vs Spectral clustering comparison for a single dataset.
    Layout: 2×3 (rows: K-means/Spectral, cols: GIN/Graph2Vec/NetLSD)
    """
    fig, axes = plt.subplots(2, 3, figsize=Config.FIGSIZE_2x3)
    
    cluster_algs = ['kmeans', 'spectral']
    cluster_labels = ['K-means', 'Spectral']
    
    for col, method in enumerate(Config.METHODS):
        # Load embeddings once (same for both clustering methods)
        best_config_km = get_best_config_for_method_dataset_dim(df, method, dataset, dim, cluster_alg='kmeans')
        
        embeddings, labels = None, None
        if best_config_km is not None:
            run_name = best_config_km.get('run_name') if pd.notna(best_config_km.get('run_name')) else None
            embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
        
        # Compute projection once
        projection = None
        if embeddings is not None:
            projection = compute_projection(embeddings, projection_type)
        
        for row, (alg, alg_label) in enumerate(zip(cluster_algs, cluster_labels)):
            ax = axes[row, col]
            
            best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim, cluster_alg=alg)
            
            if embeddings is None or best_config is None:
                plot_empty_cell(ax, 'N/A', method if row == 0 else None)
                if col == 0:
                    ax.set_ylabel(alg_label, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            metrics_text = f"ARI={best_config['ari']:.3f}\nNMI={best_config['nmi']:.3f}"
            
            title = method if row == 0 else None
            plot_embedding_scatter(ax, projection, labels, title=title,
                                  metrics_text=metrics_text, show_legend=(row == 0 and col == 2))
            
            if col == 0:
                ax.set_ylabel(alg_label, fontsize=Config.LABEL_SIZE, fontweight='bold')
    
    fig.suptitle(f"{dataset} - K-means vs Spectral ({projection_type.upper()}, dim={dim})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# NEW: COMPREHENSIVE GRID - SEPARATE FOR ORIGINAL AND PERMUTED (3×3)
# =============================================================================

def create_comprehensive_single_source(df, base_dir, datasets_dir, projection_type='tsne',
                                       dim=128, source_label='Original', output_path=None):
    """
    Comprehensive grid for a single source (original OR permuted).
    Layout: 3×3 (rows: Methods, cols: Datasets)
    """
    fig, axes = plt.subplots(3, 3, figsize=Config.FIGSIZE_3x3)
    
    for m_idx, method in enumerate(Config.METHODS):
        for d_idx, dataset in enumerate(Config.DATASETS):
            ax = axes[m_idx, d_idx]
            
            best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
            
            if best_config is None:
                plot_empty_cell(ax, 'N/A')
                if m_idx == 0:
                    ax.set_title(dataset, fontsize=Config.TITLE_SIZE, fontweight='bold')
                if d_idx == 0:
                    ax.set_ylabel(method, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
            embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
            
            if embeddings is None:
                plot_empty_cell(ax, 'Missing')
                if m_idx == 0:
                    ax.set_title(dataset, fontsize=Config.TITLE_SIZE, fontweight='bold')
                if d_idx == 0:
                    ax.set_ylabel(method, fontsize=Config.LABEL_SIZE, fontweight='bold')
                continue
            
            projection = compute_projection(embeddings, projection_type)
            metrics_text = f"ARI={best_config['ari']:.3f}"
            
            plot_embedding_scatter(ax, projection, labels, metrics_text=metrics_text,
                                  show_legend=(m_idx == 0 and d_idx == 2))
            
            if m_idx == 0:
                ax.set_title(dataset, fontsize=Config.TITLE_SIZE, fontweight='bold')
            if d_idx == 0:
                ax.set_ylabel(method, fontsize=Config.LABEL_SIZE, fontweight='bold')
    
    fig.suptitle(f"{source_label} Embeddings - Methods × Datasets ({projection_type.upper()}, dim={dim})",
                fontsize=Config.TITLE_SIZE + 1, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# NEW: t-SNE vs UMAP COMPREHENSIVE (3×3 per projection, side by side = 3×6)
# =============================================================================

def create_tsne_umap_all_methods_datasets(df, base_dir, datasets_dir, dim=128, output_path=None):
    """
    Comprehensive t-SNE vs UMAP: All methods × all datasets.
    Layout: 3×6 (rows: Methods, cols: Datasets × t-SNE/UMAP)
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available.")
        return None
    
    fig = plt.figure(figsize=Config.FIGSIZE_3x6)
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.08, hspace=0.25)
    
    for m_idx, method in enumerate(Config.METHODS):
        for d_idx, dataset in enumerate(Config.DATASETS):
            best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim)
            
            embeddings, labels = None, None
            if best_config is not None:
                run_name = best_config.get('run_name') if pd.notna(best_config.get('run_name')) else None
                embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
            
            for p_idx, proj_type in enumerate(['tsne', 'umap']):
                col = d_idx * 2 + p_idx
                ax = fig.add_subplot(gs[m_idx, col])
                
                if embeddings is None:
                    plot_empty_cell(ax, 'N/A')
                    if m_idx == 0:
                        label = 't-SNE' if proj_type == 'tsne' else 'UMAP'
                        ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(method, fontsize=8, fontweight='bold')
                    continue
                
                projection = compute_projection(embeddings, proj_type)
                metrics_text = f"ARI={best_config['ari']:.2f}"
                
                plot_embedding_scatter(ax, projection, labels, metrics_text=metrics_text, marker_size=5)
                
                if m_idx == 0:
                    label = 't-SNE' if proj_type == 'tsne' else 'UMAP'
                    ax.set_title(f"{dataset}\n({label})", fontsize=7, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(method, fontsize=8, fontweight='bold')
    
    fig.suptitle(f"t-SNE vs UMAP - All Methods × All Datasets (dim={dim})",
                fontsize=10, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# NEW: K-MEANS vs SPECTRAL COMPREHENSIVE (3×6)
# =============================================================================

def create_kmeans_spectral_all_methods_datasets(df, base_dir, datasets_dir, 
                                                 projection_type='tsne', dim=128, output_path=None):
    """
    Comprehensive K-means vs Spectral: All methods × all datasets.
    Layout: 3×6 (rows: Methods, cols: Datasets × K-means/Spectral)
    """
    fig = plt.figure(figsize=Config.FIGSIZE_3x6)
    gs = gridspec.GridSpec(3, 6, figure=fig, wspace=0.08, hspace=0.25)
    
    cluster_algs = ['kmeans', 'spectral']
    cluster_short = ['KM', 'Spec']
    
    for m_idx, method in enumerate(Config.METHODS):
        for d_idx, dataset in enumerate(Config.DATASETS):
            # Load embeddings once
            best_config_km = get_best_config_for_method_dataset_dim(df, method, dataset, dim, cluster_alg='kmeans')
            
            embeddings, labels = None, None
            if best_config_km is not None:
                run_name = best_config_km.get('run_name') if pd.notna(best_config_km.get('run_name')) else None
                embeddings, labels = load_embeddings(method, dataset, dim, base_dir, datasets_dir, run_name)
            
            # Compute projection once
            projection = None
            if embeddings is not None:
                projection = compute_projection(embeddings, projection_type)
            
            for c_idx, (alg, alg_short) in enumerate(zip(cluster_algs, cluster_short)):
                col = d_idx * 2 + c_idx
                ax = fig.add_subplot(gs[m_idx, col])
                
                best_config = get_best_config_for_method_dataset_dim(df, method, dataset, dim, cluster_alg=alg)
                
                if embeddings is None or best_config is None:
                    plot_empty_cell(ax, 'N/A')
                    if m_idx == 0:
                        ax.set_title(f"{dataset}\n({alg_short})", fontsize=7, fontweight='bold')
                    if col == 0:
                        ax.set_ylabel(method, fontsize=8, fontweight='bold')
                    continue
                
                metrics_text = f"ARI={best_config['ari']:.2f}"
                
                plot_embedding_scatter(ax, projection, labels, metrics_text=metrics_text, marker_size=5)
                
                if m_idx == 0:
                    ax.set_title(f"{dataset}\n({alg_short})", fontsize=7, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(method, fontsize=8, fontweight='bold')
    
    fig.suptitle(f"K-means vs Spectral - All Methods × All Datasets ({projection_type.upper()}, dim={dim})",
                fontsize=10, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_all_figures(base_dir, datasets_dir, permuted_base_dir=None, output_dir=None,
                        metrics_csv=None, projection_types=['tsne', 'umap']):
    """Generate all visualization figures."""
    output_dir = Path(output_dir) if output_dir else Config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_metrics(metrics_csv)
    print(f"Loaded {len(df)} records")
    print(f"Datasets: {list(df['dataset'].unique())}")
    print(f"Methods: {list(df['method'].unique())}")
    
    generated = []
    
    # Create subdirectories
    (output_dir / "single_row").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)
    (output_dir / "grids").mkdir(exist_ok=True)
    (output_dir / "tsne_vs_umap").mkdir(exist_ok=True)
    (output_dir / "kmeans_vs_spectral").mkdir(exist_ok=True)
    (output_dir / "comprehensive_original").mkdir(exist_ok=True)
    (output_dir / "comprehensive_permuted").mkdir(exist_ok=True)
    
    for proj in projection_types:
        if proj == 'umap' and not UMAP_AVAILABLE:
            continue
        
        print(f"\n=== Generating {proj.upper()} figures ===")
        
        # --- SINGLE ROW FIGURES ---
        print("Creating single-row figures...")
        for dataset in Config.DATASETS:
            fig = create_method_comparison_single(
                df, dataset, 128, base_dir, datasets_dir, proj,
                output_dir / "single_row" / f"methods_{dataset}_{proj}.png")
            if fig:
                generated.append(f"single_row/methods_{dataset}_{proj}.png")
                plt.close(fig)
        
        for method in Config.METHODS:
            fig = create_dataset_comparison_single(
                df, method, 128, base_dir, datasets_dir, proj,
                output_dir / "single_row" / f"datasets_{method}_{proj}.png")
            if fig:
                generated.append(f"single_row/datasets_{method}_{proj}.png")
                plt.close(fig)
        
        # --- METHODS × DIMENSIONS GRIDS ---
        print("Creating methods×dimensions grids...")
        for dataset in Config.DATASETS:
            fig = create_methods_dimensions_grid(
                df, dataset, base_dir, datasets_dir, proj,
                output_dir / "grids" / f"methods_dims_{dataset}_{proj}.png")
            if fig:
                generated.append(f"grids/methods_dims_{dataset}_{proj}.png")
                plt.close(fig)
        
        # --- COMPREHENSIVE ORIGINAL (3×3) ---
        print("Creating comprehensive original grid...")
        fig = create_comprehensive_single_source(
            df, base_dir, datasets_dir, proj, 128, 'Original',
            output_dir / "comprehensive_original" / f"comprehensive_original_{proj}.png")
        if fig:
            generated.append(f"comprehensive_original/comprehensive_original_{proj}.png")
            plt.close(fig)
        
        # --- COMPREHENSIVE PERMUTED (3×3) ---
        if permuted_base_dir:
            print("Creating comprehensive permuted grid...")
            fig = create_comprehensive_single_source(
                df, permuted_base_dir, datasets_dir, proj, 128, 'Permuted',
                output_dir / "comprehensive_permuted" / f"comprehensive_permuted_{proj}.png")
            if fig:
                generated.append(f"comprehensive_permuted/comprehensive_permuted_{proj}.png")
                plt.close(fig)
        
        # --- K-MEANS vs SPECTRAL PER DATASET (2×3) ---
        print("Creating K-means vs Spectral per dataset...")
        for dataset in Config.DATASETS:
            fig = create_kmeans_vs_spectral_per_dataset(
                df, dataset, 128, base_dir, datasets_dir, proj,
                output_dir / "kmeans_vs_spectral" / f"kmeans_vs_spectral_{dataset}_{proj}.png")
            if fig:
                generated.append(f"kmeans_vs_spectral/kmeans_vs_spectral_{dataset}_{proj}.png")
                plt.close(fig)
        
        # --- K-MEANS vs SPECTRAL COMPREHENSIVE (3×6) ---
        print("Creating K-means vs Spectral comprehensive...")
        fig = create_kmeans_spectral_all_methods_datasets(
            df, base_dir, datasets_dir, proj, 128,
            output_dir / "kmeans_vs_spectral" / f"kmeans_vs_spectral_comprehensive_{proj}.png")
        if fig:
            generated.append(f"kmeans_vs_spectral/kmeans_vs_spectral_comprehensive_{proj}.png")
            plt.close(fig)
        
        # --- COMPARISON FIGURES (if permuted exists) ---
        if permuted_base_dir:
            print("Creating comparison figures...")
            for dataset in Config.DATASETS:
                fig = create_method_comparison_dual(
                    df, dataset, 128, base_dir, datasets_dir, permuted_base_dir, proj,
                    output_dir / "comparison" / f"compare_methods_{dataset}_{proj}.png")
                if fig:
                    generated.append(f"comparison/compare_methods_{dataset}_{proj}.png")
                    plt.close(fig)
            
            fig = create_comprehensive_comparison_grid(
                df, base_dir, datasets_dir, permuted_base_dir, proj, 128,
                output_dir / "comparison" / f"comprehensive_orig_vs_perm_{proj}.png")
            if fig:
                generated.append(f"comparison/comprehensive_orig_vs_perm_{proj}.png")
                plt.close(fig)
    
    # --- t-SNE vs UMAP PER DATASET (2×3) ---
    if UMAP_AVAILABLE:
        print("Creating t-SNE vs UMAP per dataset...")
        for dataset in Config.DATASETS:
            fig = create_tsne_vs_umap_per_dataset(
                df, dataset, 128, base_dir, datasets_dir,
                output_dir / "tsne_vs_umap" / f"tsne_vs_umap_{dataset}.png")
            if fig:
                generated.append(f"tsne_vs_umap/tsne_vs_umap_{dataset}.png")
                plt.close(fig)
        
        # --- t-SNE vs UMAP COMPREHENSIVE (3×6) ---
        print("Creating t-SNE vs UMAP comprehensive...")
        fig = create_tsne_umap_all_methods_datasets(
            df, base_dir, datasets_dir, 128,
            output_dir / "tsne_vs_umap" / "tsne_vs_umap_comprehensive.png")
        if fig:
            generated.append("tsne_vs_umap/tsne_vs_umap_comprehensive.png")
            plt.close(fig)
        
        # --- t-SNE vs UMAP per method (existing) ---
        print("Creating t-SNE vs UMAP comparisons per method...")
        fig = create_tsne_umap_comparison_grid(
            df, base_dir, datasets_dir, 128,
            output_dir / "grids" / "tsne_vs_umap_grid.png")
        if fig:
            generated.append("grids/tsne_vs_umap_grid.png")
            plt.close(fig)
        
        for method in Config.METHODS:
            fig = create_tsne_umap_single_method(
                df, method, base_dir, datasets_dir, 128,
                output_dir / "grids" / f"tsne_vs_umap_{method}.png")
            if fig:
                generated.append(f"grids/tsne_vs_umap_{method}.png")
                plt.close(fig)
    
    # --- METRICS HEATMAPS ---
    print("Creating metrics heatmaps...")
    for alg in ['kmeans', 'spectral']:
        fig = create_metrics_heatmap(df, alg, output_dir / f"metrics_heatmap_{alg}.png")
        if fig:
            generated.append(f"metrics_heatmap_{alg}.png")
            plt.close(fig)
    
    print(f"\n=== Generated {len(generated)} figures ===")
    for f in generated:
        print(f"  - {f}")
    
    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate t-SNE/UMAP visualizations')
    parser.add_argument('--embeddings-dir', type=str, default=None)
    parser.add_argument('--permuted-dir', type=str, default=None)
    parser.add_argument('--datasets-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--metrics-csv', type=str, default=None)
    parser.add_argument('--projections', nargs='+', default=['tsne', 'umap'],
                       choices=['tsne', 'umap'])
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    base_dir = Path(args.embeddings_dir) if args.embeddings_dir else project_root / "embeddings"
    permuted_dir = Path(args.permuted_dir) if args.permuted_dir else project_root / "permutated_embeddings"
    datasets_dir = Path(args.datasets_dir) if args.datasets_dir else project_root / "DATASETS"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "plots"
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else script_dir / "clustering_merged.csv"
    
    if not permuted_dir.exists():
        print(f"Permuted directory not found: {permuted_dir}")
        permuted_dir = None
    
    if not base_dir.exists():
        print(f"Error: Embeddings directory not found: {base_dir}")
        return
    
    if not metrics_csv.exists():
        print(f"Error: Metrics CSV not found: {metrics_csv}")
        return
    
    print(f"Embeddings: {base_dir}")
    print(f"Permuted: {permuted_dir}")
    print(f"Datasets: {datasets_dir}")
    print(f"Output: {output_dir}")
    print(f"Metrics: {metrics_csv}")
    
    setup_matplotlib_style()
    generate_all_figures(base_dir, datasets_dir, permuted_dir, output_dir, metrics_csv, args.projections)


if __name__ == "__main__":
    main()