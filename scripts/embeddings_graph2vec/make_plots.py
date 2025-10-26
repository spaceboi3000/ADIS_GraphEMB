import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Configuration
class Config:
    """Centralized configuration for consistent styling."""
    STYLE = 'seaborn-v0_8'
    PALETTE = "husl"
    DPI = 300
    FIGSIZE_LARGE = (15, 12)
    FIGSIZE_MEDIUM = (14, 6)
    COLOR_FIT = '#2E86AB'
    COLOR_EMBED = '#A23B72'
    COLOR_PALETTE = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    PLOTS_DIR = Path('./plots')

# Initialize styling
plt.style.use(Config.STYLE)
sns.set_palette(Config.PALETTE)
Config.PLOTS_DIR.mkdir(exist_ok=True)


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load CSV and calculate derived metrics."""
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = ['fit_time_s', 'embed_time_s', 'total_time_s', 'dataset', 'dim', 'n_graphs']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate derived metrics
    df['fit_embed_ratio'] = df['fit_time_s'] / df['embed_time_s']
    df['embed_percentage'] = (df['embed_time_s'] / df['total_time_s']) * 100
    
    return df


# --- Enhancements for prettier graphs ---
sns.set_context("talk")  # larger font sizes
sns.set_style("whitegrid")  # clean background

def create_stacked_time_plot(df: pd.DataFrame, ax: plt.Axes, dataset: str) -> None:
    """Create a polished stacked bar chart for time components (log scale)."""
    data = df[df['dataset'] == dataset].sort_values('dim')
    x = np.arange(len(data))
    width = 0.7

    # Avoid zero values for log scale
    fit_times = data['fit_time_s'].replace(0, 1e-6)
    embed_times = data['embed_time_s'].replace(0, 1e-6)

    # Stacked bars
    ax.bar(x, fit_times, width, label='Fit Time', color="#2E86AB", alpha=0.85)
    ax.bar(x, embed_times, width, bottom=fit_times, label='Embed Time', color="#A23B72", alpha=0.85)

    # Log scale
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(data['dim'].astype(int))
    ax.set_title(f"{dataset} (n={data['n_graphs'].iloc[0]:,} graphs)", fontweight='bold', fontsize=12)
    ax.set_xlabel('Embedding Dimension', fontsize=11)
    ax.set_ylabel('Time (s, log scale)', fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.legend(framealpha=0.9)

    # Add embed time annotations (placed at top of embed bar, small offset)
    for i, (fit, embed) in enumerate(zip(fit_times, embed_times)):
        if embed > 0:
            ax.text(i, fit + embed*1.05, f'{embed:.4f}s', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color="#A23B72")

    # Set y-limits to prevent overly tall plots
    max_total = (fit_times + embed_times).max()
    ax.set_ylim(1e-6, max_total * 1.15)  # 15% padding


def plot_time_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    """Generate polished time breakdown visualization."""
    datasets = df['dataset'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16,12))
    axes = axes.flatten()

    for i, dataset in enumerate(datasets[:3]):
        create_stacked_time_plot(df, axes[i], dataset)

    axes[3].remove()
    ax_ratio = fig.add_subplot(2, 2, 4)

    # Plot fit/embed ratios
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('dim')
        ax_ratio.plot(data['dim'], data['fit_embed_ratio'], marker='o', linewidth=2.5, markersize=8,
                      label=dataset, alpha=0.85)
    
    ax_ratio.set_yscale('log')
    ax_ratio.set_xlabel('Embedding Dimension', fontsize=11)
    ax_ratio.set_ylabel('Fit / Embed Time Ratio (log)', fontsize=11)
    ax_ratio.set_title('Time Ratio Comparison', fontsize=12, fontweight='bold')
    ax_ratio.grid(True, linestyle='--', alpha=0.3)
    ax_ratio.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def plot_embed_time_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Polished embedding time analysis with dynamic y-axis scaling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGSIZE_MEDIUM)
    
    # Left: Embed times in milliseconds
    max_val = 0
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('dim')
        embed_ms = data['embed_time_s'] * 1000
        ax1.plot(data['dim'], embed_ms, marker='s', linewidth=2.5,
                markersize=8, label=dataset, alpha=0.85)
        
        # Track max value for axis limit
        max_val = max(max_val, embed_ms.max())
        
        # Annotate only the max dimension
        max_row = data.iloc[-1]
        ax1.annotate(f'{max_row["embed_time_s"]*1000:.1f}ms',
                    xy=(max_row['dim'], max_row['embed_time_s']*1000),
                    xytext=(5, 2), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    # Set y-axis limit with appropriate padding
    ax1.set_ylim(0, max_val * 1.12)
    ax1.set_xlabel('Embedding Dimension', fontsize=11)
    ax1.set_ylabel('Embed Time (milliseconds)', fontsize=11)
    ax1.set_title('Embedding Generation Time', fontweight='bold', fontsize=12)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Right: Percentage contribution with dynamic scaling
    labels = [f"{row['dataset']}\nDim {row['dim']}" for _, row in df.iterrows()]
    x_pos = np.arange(len(df))
    
    bars = ax2.bar(x_pos, df['embed_percentage'], 
                   color=Config.COLOR_PALETTE * (len(df) // len(Config.COLOR_PALETTE) + 1),
                   alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Calculate appropriate y-axis limit based on data range
    max_pct = df['embed_percentage'].max()
    y_limit = max_pct * 1.15  # 15% padding for labels
    
    ax2.set_ylim(0, y_limit)
    ax2.set_xlabel('Experiment Configuration', fontsize=11)
    ax2.set_ylabel('Embed Time (% of Total)', fontsize=11)
    ax2.set_title('Embed Time as % of Total Runtime', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # Add percentage labels with dynamic offset
    for bar, percentage in zip(bars, df['embed_percentage']):
        height = bar.get_height()
        # Dynamic offset: 2% of the y-axis limit
        offset = y_limit * 0.02
        ax2.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{percentage:.3f}%', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()

def plot_performance_heatmaps(df: pd.DataFrame, output_path: Path) -> None:
    """Polished performance heatmaps."""
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    
    time_pivot = df.pivot(index='dataset', columns='dim', values='total_time_s')
    ratio_pivot = df.pivot(index='dataset', columns='dim', values='fit_embed_ratio')

    sns.heatmap(time_pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[0], linewidths=0.5)
    axes[0].set_title('Total Execution Time (s)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Embedding Dimension', fontsize=11)
    axes[0].set_ylabel('Dataset', fontsize=11)

    sns.heatmap(ratio_pivot, annot=True, fmt=".1f", cmap="RdPu", ax=axes[1], linewidths=0.5)
    axes[1].set_title('Fit / Embed Time Ratio', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Embedding Dimension', fontsize=11)
    axes[1].set_ylabel('Dataset', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    try:
        # Load and prepare data
        df = load_and_prepare_data('metrics_graph2vec.csv')
        
        # Generate all plots
        plots = [
            ('time_breakdown_clean.png', plot_time_breakdown),
            ('embed_time_analysis.png', plot_embed_time_analysis),
            ('performance_heatmaps_clean.png', plot_performance_heatmaps)
        ]
        
        print(f"Generating plots in '{Config.PLOTS_DIR}'...\n")
        
        for filename, plot_func in plots:
            output_path = Config.PLOTS_DIR / filename
            plot_func(df, output_path)
            print(f"✓ {filename} - {plot_func.__doc__}")
        
        print(f"\n✅ All plots successfully saved to '{Config.PLOTS_DIR}/'")
        
        # Print summary statistics
        print("\n📊 Dataset Summary:")
        print(df.groupby('dataset').agg({
            'n_graphs': 'first',
            'total_time_s': 'mean',
            'embed_percentage': 'mean'
        }).round(3))
        
    except FileNotFoundError:
        print("❌ Error: 'metrics_graph2vec.csv' not found in current directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()