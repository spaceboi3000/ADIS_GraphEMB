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

def plot_memory_usage_per_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Plot memory usage per dataset per method with before/after comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGSIZE_MEDIUM)
    
    # Left: Memory before and after
    datasets = df['dataset'].unique()
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    # Calculate memory values
    before_memory = [df[df['dataset'] == dataset]['rss_before_mb'].mean() for dataset in datasets]
    after_memory = [df[df['dataset'] == dataset]['rss_after_mb'].mean() for dataset in datasets]
    
    bars1 = ax1.bar(x_pos - width/2, before_memory, width, label='Before Execution', 
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x_pos + width/2, after_memory, width, label='After Execution', 
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Dataset', fontsize=11)
    ax1.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax1.set_title('Memory Usage: Before vs After Execution', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45, ha='right')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}MB', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right: Memory increase percentage
    memory_increase = [(after - before) / before * 100 for before, after in zip(before_memory, after_memory)]
    
    colors = ['#FF6B6B' if inc > 0 else '#4ECDC4' for inc in memory_increase]
    bars3 = ax2.bar(x_pos, memory_increase, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Dataset', fontsize=11)
    ax2.set_ylabel('Memory Increase (%)', fontsize=11)
    ax2.set_title('Memory Usage Increase Percentage', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for bar, percentage in zip(bars3, memory_increase):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -2),
                f'{percentage:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    # Add zero reference line
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def plot_memory_vs_dimension(df: pd.DataFrame, output_path: Path) -> None:
    """Plot memory usage vs embedding dimension for each dataset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIGSIZE_MEDIUM)
    
    # Left: Peak memory vs dimension
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('dim')
        ax1.plot(data['dim'], data['peak_tracemalloc_mb'], marker='o', linewidth=2.5,
                markersize=8, label=dataset, alpha=0.85)
        
        # Annotate last point
        last_row = data.iloc[-1]
        ax1.annotate(f'{last_row["peak_tracemalloc_mb"]:.0f}MB',
                    xy=(last_row['dim'], last_row['peak_tracemalloc_mb']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Embedding Dimension', fontsize=11)
    ax1.set_ylabel('Peak Memory Usage (MB)', fontsize=11)
    ax1.set_title('Peak Memory vs Embedding Dimension', fontweight='bold', fontsize=12)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Right: Memory efficiency (memory per graph)
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('dim')
        memory_per_graph = data['peak_tracemalloc_mb'] / data['n_graphs']
        ax2.plot(data['dim'], memory_per_graph, marker='s', linewidth=2.5,
                markersize=8, label=dataset, alpha=0.85)
        
        # Annotate last point
        last_row = data.iloc[-1]
        efficiency = last_row['peak_tracemalloc_mb'] / last_row['n_graphs']
        ax2.annotate(f'{efficiency:.2f}MB/graph',
                    xy=(last_row['dim'], efficiency),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Embedding Dimension', fontsize=11)
    ax2.set_ylabel('Memory per Graph (MB)', fontsize=11)
    ax2.set_title('Memory Efficiency: Peak Memory per Graph', fontweight='bold', fontsize=12)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def plot_comprehensive_analysis(df: pd.DataFrame, output_path: Path) -> None:
    """Comprehensive analysis showing relationships between time, memory, and dimensions."""
    fig, axes = plt.subplots(2, 2, figsize=Config.FIGSIZE_LARGE)
    
    # Top-left: Time vs Memory
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('dim')
        scatter = axes[0,0].scatter(data['total_time_s'], data['peak_tracemalloc_mb'], 
                                   s=data['dim']*20, alpha=0.7, label=dataset)
        
        # Add dimension labels for some points
        for i, row in data.iterrows():
            if row['dim'] in [64, 128, 256]:  # Label only some dimensions for clarity
                axes[0,0].annotate(f"Dim {row['dim']}", 
                                 (row['total_time_s'], row['peak_tracemalloc_mb']),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8, alpha=0.8)
    
    axes[0,0].set_xlabel('Total Time (s)', fontsize=11)
    axes[0,0].set_ylabel('Peak Memory (MB)', fontsize=11)
    axes[0,0].set_title('Time vs Memory Usage', fontweight='bold', fontsize=12)
    axes[0,0].legend(framealpha=0.9)
    axes[0,0].grid(True, alpha=0.3)
    
    # Top-right: Memory usage breakdown per dataset
    memory_metrics = ['rss_before_mb', 'rss_after_mb', 'peak_tracemalloc_mb']
    memory_data = df.groupby('dataset')[memory_metrics].mean()
    
    x_pos = np.arange(len(memory_data))
    width = 0.25
    
    for i, metric in enumerate(memory_metrics):
        axes[0,1].bar(x_pos + i*width, memory_data[metric], width, 
                     label=metric.replace('_mb', '').replace('_', ' ').title(),
                     alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[0,1].set_xlabel('Dataset', fontsize=11)
    axes[0,1].set_ylabel('Memory (MB)', fontsize=11)
    axes[0,1].set_title('Memory Usage Breakdown', fontweight='bold', fontsize=12)
    axes[0,1].set_xticks(x_pos + width)
    axes[0,1].set_xticklabels(memory_data.index, rotation=45, ha='right')
    axes[0,1].legend(framealpha=0.9)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # Bottom-left: Time per graph vs dimension
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('dim')
        time_per_graph = data['total_time_s'] / data['n_graphs']
        axes[1,0].plot(data['dim'], time_per_graph, marker='o', linewidth=2.5,
                      markersize=8, label=dataset, alpha=0.85)
    
    axes[1,0].set_xlabel('Embedding Dimension', fontsize=11)
    axes[1,0].set_ylabel('Time per Graph (s)', fontsize=11)
    axes[1,0].set_title('Computational Efficiency: Time per Graph', fontweight='bold', fontsize=12)
    axes[1,0].legend(framealpha=0.9)
    axes[1,0].grid(True, alpha=0.3, linestyle='--')
    
    # Bottom-right: Memory vs number of graphs
    scatter = axes[1,1].scatter(df['n_graphs'], df['peak_tracemalloc_mb'], 
                               c=df['dim'], s=100, alpha=0.7, cmap='viridis')
    
    axes[1,1].set_xlabel('Number of Graphs', fontsize=11)
    axes[1,1].set_ylabel('Peak Memory (MB)', fontsize=11)
    axes[1,1].set_title('Memory vs Dataset Size (color=dimension)', fontweight='bold', fontsize=12)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add colorbar for dimension
    cbar = plt.colorbar(scatter, ax=axes[1,1])
    cbar.set_label('Embedding Dimension', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def plot_performance_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Create a comprehensive performance summary dashboard."""
    fig = plt.figure(figsize=(18, 14))
    
    # Create subplot grid
    gs = fig.add_gridspec(3, 3)
    
    # 1. Total time by dataset and dimension (heatmap style)
    ax1 = fig.add_subplot(gs[0, 0])
    time_pivot = df.pivot(index='dataset', columns='dim', values='total_time_s')
    sns.heatmap(time_pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax1, linewidths=0.5)
    ax1.set_title('Total Time (s) by Dataset & Dimension', fontweight='bold', fontsize=11)
    
    # 2. Memory usage by dataset and dimension
    ax2 = fig.add_subplot(gs[0, 1])
    memory_pivot = df.pivot(index='dataset', columns='dim', values='peak_tracemalloc_mb')
    sns.heatmap(memory_pivot, annot=True, fmt=".0f", cmap="Blues", ax=ax2, linewidths=0.5)
    ax2.set_title('Peak Memory (MB) by Dataset & Dimension', fontweight='bold', fontsize=11)
    
    # 3. Time ratio analysis
    ax3 = fig.add_subplot(gs[0, 2])
    ratio_pivot = df.pivot(index='dataset', columns='dim', values='fit_embed_ratio')
    sns.heatmap(ratio_pivot, annot=True, fmt=".1f", cmap="RdPu", ax=ax3, linewidths=0.5)
    ax3.set_title('Fit/Embed Time Ratio', fontweight='bold', fontsize=11)
    
    # 4. Memory efficiency scatter
    ax4 = fig.add_subplot(gs[1, :])
    colors = plt.cm.Set1(np.linspace(0, 1, len(df['dataset'].unique())))
    
    for i, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        ax4.scatter(data['total_time_s'], data['peak_tracemalloc_mb'], 
                   s=data['dim']*10, alpha=0.7, label=dataset, color=colors[i])
        
        # Add dataset labels
        for _, row in data.iterrows():
            ax4.annotate(f"Dim {row['dim']}", 
                        (row['total_time_s'], row['peak_tracemalloc_mb']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    ax4.set_xlabel('Total Time (s)', fontsize=11)
    ax4.set_ylabel('Peak Memory (MB)', fontsize=11)
    ax4.set_title('Performance Overview: Time vs Memory', fontweight='bold', fontsize=12)
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Memory comparison bars
    ax5 = fig.add_subplot(gs[2, 0])
    memory_comparison = df[['dataset', 'rss_before_mb', 'rss_after_mb', 'peak_tracemalloc_mb']].copy()
    memory_comparison = memory_comparison.groupby('dataset').mean()
    
    x_pos = np.arange(len(memory_comparison))
    width = 0.25
    
    ax5.bar(x_pos - width, memory_comparison['rss_before_mb'], width, 
            label='Before', alpha=0.8, color='#2E86AB')
    ax5.bar(x_pos, memory_comparison['rss_after_mb'], width, 
            label='After', alpha=0.8, color='#A23B72')
    ax5.bar(x_pos + width, memory_comparison['peak_tracemalloc_mb'], width, 
            label='Peak', alpha=0.8, color='#FF6B6B')
    
    ax5.set_xlabel('Dataset', fontsize=11)
    ax5.set_ylabel('Memory (MB)', fontsize=11)
    ax5.set_title('Memory Usage Comparison', fontweight='bold', fontsize=11)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(memory_comparison.index, rotation=45, ha='right')
    ax5.legend(framealpha=0.9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Efficiency metrics
    ax6 = fig.add_subplot(gs[2, 1])
    efficiency_data = []
    for _, row in df.iterrows():
        efficiency_data.append({
            'dataset': row['dataset'],
            'time_per_graph': row['total_time_s'] / row['n_graphs'],
            'memory_per_graph': row['peak_tracemalloc_mb'] / row['n_graphs'],
            'dim': row['dim']
        })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    
    for dataset in efficiency_df['dataset'].unique():
        data = efficiency_df[efficiency_df['dataset'] == dataset].sort_values('dim')
        ax6.plot(data['dim'], data['time_per_graph'], marker='o', linewidth=2,
                label=dataset, alpha=0.85)
    
    ax6.set_xlabel('Embedding Dimension', fontsize=11)
    ax6.set_ylabel('Time per Graph (s)', fontsize=11)
    ax6.set_title('Time Efficiency per Graph', fontweight='bold', fontsize=11)
    ax6.legend(framealpha=0.9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Memory increase percentage
    ax7 = fig.add_subplot(gs[2, 2])
    memory_increase = []
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        avg_before = data['rss_before_mb'].mean()
        avg_after = data['rss_after_mb'].mean()
        increase_pct = (avg_after - avg_before) / avg_before * 100
        memory_increase.append({'dataset': dataset, 'increase_pct': increase_pct})
    
    increase_df = pd.DataFrame(memory_increase)
    
    colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in increase_df['increase_pct']]
    bars = ax7.bar(increase_df['dataset'], increase_df['increase_pct'], color=colors, alpha=0.8)
    
    ax7.set_xlabel('Dataset', fontsize=11)
    ax7.set_ylabel('Memory Increase (%)', fontsize=11)
    ax7.set_title('Memory Usage Increase', fontweight='bold', fontsize=11)
    ax7.set_xticklabels(increase_df['dataset'], rotation=45, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add percentage labels
    for bar, percentage in zip(bars, increase_df['increase_pct']):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -2),
                f'{percentage:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=Config.DPI, bbox_inches='tight')
    plt.close()


# Update the main function to include the new plots
def main():
    """Main execution function."""
    try:
        # Load and prepare data
        df = load_and_prepare_data('metrics_graph2vec.csv')
        
        # Generate all plots (original + new)
        plots = [
            ('time_breakdown_clean.png', plot_time_breakdown),
            ('embed_time_analysis.png', plot_embed_time_analysis),
            ('performance_heatmaps_clean.png', plot_performance_heatmaps),
            ('memory_usage_per_dataset.png', plot_memory_usage_per_dataset),
            ('memory_vs_dimension.png', plot_memory_vs_dimension),
            ('comprehensive_analysis.png', plot_comprehensive_analysis),
            ('performance_summary_dashboard.png', plot_performance_summary)
        ]
        
        print(f"Generating plots in '{Config.PLOTS_DIR}'...\n")
        
        for filename, plot_func in plots:
            output_path = Config.PLOTS_DIR / filename
            plot_func(df, output_path)
            print(f"✓ {filename} - {plot_func.__doc__.split('.')[0]}")
        
        print(f"\n✅ All plots successfully saved to '{Config.PLOTS_DIR}/'")
        
        # Print summary statistics
        print("\n📊 Dataset Summary:")
        summary = df.groupby('dataset').agg({
            'n_graphs': 'first',
            'total_time_s': 'mean',
            'embed_percentage': 'mean',
            'rss_before_mb': 'mean',
            'rss_after_mb': 'mean',
            'peak_tracemalloc_mb': 'mean'
        }).round(3)
        print(summary)
        
        # Print memory statistics
        print("\n💾 Memory Statistics:")
        df['memory_increase_mb'] = df['rss_after_mb'] - df['rss_before_mb']
        df['memory_increase_pct'] = (df['memory_increase_mb'] / df['rss_before_mb']) * 100
        
        memory_stats = df.groupby('dataset').agg({
            'memory_increase_mb': 'mean',
            'memory_increase_pct': 'mean',
            'peak_tracemalloc_mb': 'mean'
        }).round(3)
        print(memory_stats)
        
    except FileNotFoundError:
        print("❌ Error: 'metrics_graph2vec.csv' not found in current directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
