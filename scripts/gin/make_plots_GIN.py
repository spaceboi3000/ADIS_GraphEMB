# new_plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


class Config:
    STYLE = 'seaborn-v0_8'
    PALETTE = "husl"
    DPI = 300
    FIGSIZE = (10, 10)
    COLOR_FIT = '#2E86AB'
    COLOR_EMBED = '#A23B72'
    COLOR_PEAK = '#FF6B6B'
    BASE = Path(__file__).parent
    OUT_DIR = BASE / "embeddings_gin_version1"
    CSV = OUT_DIR / "metrics_gin.csv"
    PLOTS_DIR = OUT_DIR / "plots_GIN"

plt.style.use(Config.STYLE)
sns.set_palette(Config.PALETTE)
sns.set_context("talk")
sns.set_style("whitegrid")
Config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_REQUIRED = {
    "fit_time_s","embed_time_s","total_time_s",
    "test_acc","test_f1_macro","test_auc_macro",
    "batch_size","epochs","lr",
    "rss_before_mb","rss_after_mb","dim","n_graphs"
}

def load_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["dataset", "name", "data", "graph_dataset"]:
            rename_map[c] = "dataset"
        elif cl in ["peak_tracemallov_mb", "peak_tracemalloc", "peak_tracemalloc_mb"]:
            rename_map[c] = "peak_tracemalloc_mb"
    df = df.rename(columns=rename_map)

    if "dataset" not in df.columns:
        raise ValueError(f" Could not find a dataset column in {path.name}. Found: {list(df.columns)}")

    if "peak_tracemalloc_mb" not in df.columns:
        print("⚠️  'peak_tracemalloc_mb' not found — creating zeros.")
        df["peak_tracemalloc_mb"] = 0.0

    missing = BASE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f" Missing required columns in CSV: {missing}")

    df["fit_embed_ratio"] = (df["fit_time_s"] / df["embed_time_s"]).replace([np.inf, -np.inf], np.nan)
    df["embed_percentage"] = (df["embed_time_s"] / df["total_time_s"]) * 100
    df["dataset"] = df["dataset"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    df["dataset"] = df["dataset"].str.replace("EMDB-MULTI", "IMDB-MULTI", regex=False)
    return df

def best_per_dim(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    d = df[df["dataset"] == dataset].dropna(subset=[metric]).copy()
    if d.empty:
        return d
    idx = d.groupby("dim")[metric].idxmax()
    return d.loc[idx].sort_values("dim")

def best_overall(df: pd.DataFrame, dataset: str, metric: str) -> pd.Series | None:
    d = df[df["dataset"] == dataset].dropna(subset=[metric])
    if d.empty:
        return None
    return d.loc[d[metric].idxmax()]


def line_vs_dim(df: pd.DataFrame, metric: str, label: str, fname: str):
    fig = plt.figure(figsize=Config.FIGSIZE)
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, metric)
        if b.empty:
            continue
        plt.plot(b["dim"], b[metric], marker="o", linewidth=2, label=ds)
    plt.xlabel("Embedding Dimension")
    plt.ylabel(label)
    plt.title(f"GIN — {label} vs Embedding Dimension")
    plt.grid(True, alpha=0.3); plt.legend()
    out = Config.PLOTS_DIR / fname
    plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
    print("Saved:", out)

def time_breakdown(df: pd.DataFrame):
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        x = np.arange(len(b))
        fig = plt.figure(figsize=Config.FIGSIZE)
        plt.bar(x, b["fit_time_s"], label="Train", color=Config.COLOR_FIT, alpha=0.9)
        plt.bar(x, b["embed_time_s"], bottom=b["fit_time_s"], label="Embed", color=Config.COLOR_EMBED, alpha=0.9)
        plt.xticks(x, b["dim"].astype(int))
        plt.yscale("log")
        plt.xlabel("Embedding Dimension"); plt.ylabel("Time (s, log)")
        plt.title(f"{ds} — GIN Time Breakdown")
        plt.legend(); plt.grid(True, which="both", alpha=0.3)
        out = Config.PLOTS_DIR / f"gin_time_breakdown_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

def memory_plots(df: pd.DataFrame):
    """Show memory usage with grouped bars + dedicated bar for peak tracemalloc."""
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        x = np.arange(len(b))
        w = 0.25
        fig = plt.figure(figsize=Config.FIGSIZE)
        plt.bar(x - w, b["rss_before_mb"], width=w, label="Before", color="#4ECDC4")
        plt.bar(x, b["rss_after_mb"], width=w, label="After", color="#2E86AB")
        plt.bar(x + w, b["peak_tracemalloc_mb"], width=w, label="Peak Tracemalloc", color=Config.COLOR_PEAK)
        plt.xticks(x, b["dim"].astype(int))
        plt.xlabel("Embedding Dimension"); plt.ylabel("Memory (MB)")
        plt.title(f"{ds} — Memory Usage vs Dimension")
        plt.legend(framealpha=0.9); plt.grid(True, axis="y", alpha=0.3)
        out = Config.PLOTS_DIR / f"gin_memory_bars_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)


    fig = plt.figure(figsize=Config.FIGSIZE)
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        plt.bar(b["dim"] + np.random.uniform(-2, 2, size=len(b)), b["peak_tracemalloc_mb"],
                width=10, alpha=0.7, label=ds)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Peak Tracemalloc (MB)")
    plt.title("GIN — Peak Memory Usage (Bar Plot per Dataset)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = Config.PLOTS_DIR / "gin_peak_tracemalloc_barplot.png"
    plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
    print("Saved:", out)

def tradeoff_dual_axis(df: pd.DataFrame):
    """Accuracy bars + total time line per dataset."""
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        dims = b["dim"].astype(int).values
        acc = b["test_acc"].values
        tsec = b["total_time_s"].values
        x = np.arange(len(dims))

        fig, ax1 = plt.subplots(figsize=Config.FIGSIZE)
        ax1.bar(x, acc, color="#4ECDC4", alpha=0.85, label="Accuracy")
        ax1.set_xlabel("Embedding Dimension")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, max(1.0, acc.max()*1.05))
        ax1.set_xticks(x); ax1.set_xticklabels(dims)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, tsec, marker="o", color="#FF6B6B", linewidth=2, label="Total Time (s)")
        ax2.set_ylabel("Total Time (s)")

        lines, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            lines += h; labels += l
        ax1.legend(lines, labels, loc="best")

        plt.title(f"{ds} — Accuracy vs Compute Cost")
        out = Config.PLOTS_DIR / f"gin_tradeoff_dualaxis_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

def efficiency_plot(df: pd.DataFrame):
    """Accuracy per second vs embedding dim."""
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        dims = b["dim"].astype(int).values
        eff = (b["test_acc"] / b["total_time_s"]).replace([np.inf, -np.inf], np.nan).values
        fig = plt.figure(figsize=Config.FIGSIZE)
        plt.plot(dims, eff, marker="o", linewidth=2, color="#45B7D1")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Accuracy per Second")
        plt.title(f"{ds} — Efficiency (Accuracy/Second)")
        plt.grid(True, alpha=0.3)
        out = Config.PLOTS_DIR / f"gin_efficiency_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

def tradeoff_heatmap(df: pd.DataFrame):
    """Heatmap: Accuracy, F1, Total time normalized."""
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        tab = pd.DataFrame({
            "Accuracy": b["test_acc"].values,
            "F1-macro": b["test_f1_macro"].values,
            "Total Time (s)": b["total_time_s"].values
        }, index=b["dim"].astype(int))
        norm = (tab - tab.min()) / (tab.max() - tab.min() + 1e-12)
        fig = plt.figure(figsize=Config.FIGSIZE)
        sns.heatmap(norm, annot=tab.round(3), fmt="", cmap="YlGnBu", cbar=False)
        plt.title(f"{ds} — Tradeoff Heatmap")
        plt.xlabel("Metric"); plt.ylabel("Embedding Dimension")
        out = Config.PLOTS_DIR / f"gin_tradeoff_heatmap_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)


def epochs_lr_heatmaps_subplots(df: pd.DataFrame):
    """
    One (20,20) figure with subplots: each subplot is a dataset.
    Cell value = BEST test_acc over (batch_size, dim) for that (epochs, lr).
    Title of each subplot includes the dataset's overall best (dim, batch_size, epochs, lr, acc).
    The best (epochs, lr) cell is highlighted with a star.
    """
    datasets = sorted(df["dataset"].unique())
    lrs = sorted(df["lr"].dropna().unique())
    epochs_list = sorted(df["epochs"].dropna().unique())
    if len(lrs) == 0 or len(epochs_list) == 0:
        print("⚠️ No lr/epochs grid to plot.")
        return

    n = len(datasets)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 15), squeeze=False)  

    for idx, ds in enumerate(datasets):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub = df[df["dataset"] == ds]

        # Build heatmap grid: best test_acc over (batch_size, dim) for each (epochs, lr)
        grid = np.full((len(epochs_list), len(lrs)), np.nan)
        for i, ep in enumerate(epochs_list):
            for j, lr in enumerate(lrs):
                cand = sub[(sub["epochs"] == ep) & (sub["lr"] == lr)]
                if not cand.empty:
                    grid[i, j] = cand["test_acc"].max()

        
        hm = sns.heatmap(
            grid, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[f"{x:g}" for x in lrs],
            yticklabels=[int(e) for e in epochs_list],
            ax=ax, cbar=(idx == 0)  # show one colorbar on the first subplot
        )
        ax.set_xlabel("lr")
        ax.set_ylabel("epochs")

        # ---- Find dataset-level overall best row (across ALL dims/bs/epochs/lr) ----
        if not sub.empty and "test_acc" in sub.columns:
            best_row = sub.loc[sub["test_acc"].idxmax()]
            best_dim = int(best_row["dim"])
            best_bs = int(best_row["batch_size"])
            best_ep = int(best_row["epochs"])
            best_lr = float(best_row["lr"])
            best_acc = float(best_row["test_acc"])

            # Title with best info (dim & batch size requested)
            ax.set_title(
                f"{ds}\nBest: dim={best_dim}, bs={best_bs}, ep={best_ep}, lr={best_lr:g}, acc={best_acc:.3f}",
                fontsize=15
            )

            # Highlight the cell of that best (epochs, lr) with a star
            # Map best epochs/lr to indices in the grid (if present)
            if best_ep in epochs_list and best_lr in lrs:
                i = epochs_list.index(best_ep)
                j = lrs.index(best_lr)
                ax.scatter(j + 0.5, i + 0.5, marker="*", s=250, color="white",
                           edgecolor="black", linewidths=0.8, zorder=3)

        else:
            ax.set_title(f"{ds}", fontsize=12)

    # Hide any unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle("GIN — Test Accuracy by (epochs, lr)\n(best over batch size & dim per dataset)",
                 y=0.98, fontsize=25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = Config.PLOTS_DIR / "gin_epochs_lr_heatmaps_all_datasets.png"
    fig.savefig(out, dpi=Config.DPI)
    plt.close(fig)
    print("Saved:", out)


def summary_csv(df: pd.DataFrame):
    rows = []
    for ds in sorted(df["dataset"].unique()):
        r = best_overall(df, ds, "test_acc")
        if r is None:
            continue
        rows.append({
            "dataset": ds,
            "dim": int(r["dim"]),
            "epochs": int(r["epochs"]),
            "batch_size": int(r["batch_size"]),
            "lr": r["lr"],
            "val_acc": r["val_acc"],
            "test_acc": r["test_acc"],
            "f1_macro": r["test_f1_macro"],
            "auc_macro": r["test_auc_macro"],
            "total_time_s": r["total_time_s"],
            "peak_tracemalloc_mb": r["peak_tracemalloc_mb"],
        })
    tbl = pd.DataFrame(rows).sort_values("dataset")
    out = Config.PLOTS_DIR / "best_gin_summary.csv"
    tbl.to_csv(out, index=False)
    print("Saved summary CSV ->", out)
    print(tbl.to_string(index=False))


def main():
    print("Reading:", Config.CSV)
    df = load_prepare(Config.CSV)

    # Performance vs dimension
    line_vs_dim(df, "test_acc", "Test Accuracy", "gin_test_acc_vs_dim.png")
    line_vs_dim(df, "test_f1_macro", "F1-macro", "gin_test_f1_macro_vs_dim.png")
    line_vs_dim(df, "test_auc_macro", "AUC (macro)", "gin_test_auc_macro_vs_dim.png")

    # Time, memory, tradeoffs
    time_breakdown(df)
    memory_plots(df)
    tradeoff_dual_axis(df)
    efficiency_plot(df)
    tradeoff_heatmap(df)

    # NEW: all datasets epochs×lr in one figure
    epochs_lr_heatmaps_subplots(df)

    # Best config summary
    summary_csv(df)

    print(f"\n All plots saved to: {Config.PLOTS_DIR}")

if __name__ == "__main__":
    main()
