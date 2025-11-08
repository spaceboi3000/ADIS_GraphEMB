# plot_best_summary_alt.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# ---------- Config ----------
FIGSIZE = (16, 16)
DPI = 300
STYLE = "seaborn-v0_8"
PALETTE = "husl"

plt.style.use(STYLE)
sns.set_palette(PALETTE)
sns.set_context("talk")
sns.set_style("whitegrid")

CANDIDATES = [
    Path(__file__).parent / "embeddings_gin_version1" / "plots_GIN" / "best_gin_summary.csv",
    Path(__file__).parent / "best_gin_summary.csv",
]

def find_csv():
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("best_gin_summary.csv not found in:\n" + "\n".join(map(str, CANDIDATES)))

def load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"dataset","dim","epochs","batch_size","lr",
                "val_acc","test_acc","f1_macro","total_time_s","peak_tracemalloc_mb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in best_gin_summary.csv: {missing}")

    if "auc_macro" not in df.columns:
        df["auc_macro"] = np.nan

    # clean names
    df["dataset"] = (df["dataset"].astype(str)
                     .str.strip()
                     .str.replace(r"\s+","",regex=True)
                     .str.replace("EMDB-MULTI","IMDB-MULTI",regex=False))

    # include batch size in label for clarity
    df["label"] = df["dataset"] + "  (bs=" + df["batch_size"].astype(int).astype(str) + ")"

    # sort by dim desc to make bars nice
    df = df.sort_values("dim", ascending=False)
    return df

def fmt_sci(x):
    """Format LR in nice scientific notation."""
    if x <= 0 or pd.isna(x):
        return "—"
    exp = int(np.floor(np.log10(x)))
    base = x / (10**exp)
    return f"{base:.1f}e{exp:+d}"

def plot_hparams_alt(df: pd.DataFrame, out_dir: Path):
    """
    Three-panel horizontal bar dashboard:
    - Embedding Dimension
    - Epochs
    - Learning Rate (log scale on x)
    Shared y-axis = datasets (with batch size in label).
    """
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)

    y = np.arange(len(df))
    labels = df["label"].tolist()

    # Panel A: Dim
    ax = axes[0]
    ax.barh(y, df["dim"], color="#4ECDC4", alpha=0.9)
    for i, v in enumerate(df["dim"]):
        ax.text(v, i, f" {int(v)}", va="center", ha="left", fontsize=10, weight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()  # largest on top (since we sorted desc)
    ax.set_xlabel("Embedding Dimension")
    ax.set_title("Embedding Dim")

    # Panel B: Epochs
    ax = axes[1]
    ax.barh(y, df["epochs"], color="#45B7D1", alpha=0.9)
    for i, v in enumerate(df["epochs"]):
        ax.text(v, i, f" {int(v)}", va="center", ha="left", fontsize=10, weight="bold")
    ax.set_xlabel("Epochs")
    ax.set_title("Training Epochs")

    # Panel C: Learning Rate (log x)
    ax = axes[2]
    # Avoid zeros for log; replace with tiny positive to render but annotate real value
    lr_safe = df["lr"].clip(lower=1e-9)
    bars = ax.barh(y, lr_safe, color="#FF6B6B", alpha=0.9)
    ax.set_xscale("log")
    for i, (bar, v) in enumerate(zip(bars, df["lr"])):
        ax.text(bar.get_width(), i, f" {fmt_sci(v)}", va="center", ha="left", fontsize=10, weight="bold")
    ax.set_xlabel("Learning Rate (log)")
    ax.set_title("Learning Rate")

    # Cosmetics
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("GIN — Best Config Hyperparameters (Clear View)", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out = out_dir / "gin_summary_hyperparams_alt.png"
    plt.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)


def _dataset_order(df: pd.DataFrame) -> list[str]:
    # stable, readable order
    preferred = ["MUTAG", "ENZYMES", "IMDB-MULTI", "REDDIT-MULTI-12K", "NCI1"]
    present = [d for d in preferred if d in df["dataset"].unique()]
    # append any others not in preferred
    rest = [d for d in df["dataset"].unique() if d not in present]
    return present + rest

def plot_runtime_memory_combo(df: pd.DataFrame, out_dir: Path):
    """
    Dual-axis chart per dataset:
      - Bars = Peak Tracemalloc (MB)
      - Line = Total Time (s)
    Reads from best_gin_summary.csv columns: dataset, total_time_s, peak_tracemalloc_mb
    """
    need = {"dataset", "total_time_s", "peak_tracemalloc_mb"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for runtime/memory plot: {missing}")

    order = _dataset_order(df)
    d = df.set_index("dataset").loc[order].reset_index()

    x = np.arange(len(d))
    fig, ax1 = plt.subplots(figsize=(12, 7))

    bars = ax1.bar(x, d["peak_tracemalloc_mb"], color="#FF6B6B", alpha=0.9, label="Peak Memory (MB)")
    ax1.set_ylabel("Peak Memory (MB)")
    ax1.set_xlabel("Dataset")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["dataset"], rotation=0)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, d["total_time_s"], marker="o", linewidth=2, color="#2E86AB", label="Total Time (s)")
    ax2.set_ylabel("Total Time (s)")

    # Value labels
    for bar, mb in zip(bars, d["peak_tracemalloc_mb"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{mb:.0f}", ha="center", va="bottom", fontsize=10)
    for xi, sec in zip(x, d["total_time_s"]):
        ax2.annotate(f"{sec:.1f}s", (xi, sec), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=10)

    # Single legend combining both axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left", frameon=True)

    plt.title("GIN — Runtime vs Peak Memory (Best per Dataset)")
    fig.tight_layout()
    out = out_dir / "gin_summary_best_runtime_peakmem.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)

def plot_best_performance_bars(df: pd.DataFrame, out_dir: Path):
    """
    Bar chart per dataset for: Test Accuracy, F1-macro, AUC-macro.
    Reads from best_gin_summary.csv columns: dataset, test_acc, f1_macro, auc_macro
    """
    # Ensure required columns
    cols = {"dataset", "test_acc", "f1_macro", "auc_macro"}
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for performance bars: {missing}")

    order = _dataset_order(df)
    d = df.set_index("dataset").loc[order].reset_index()

    # Melt to long format for grouped bars
    long = d.melt(id_vars=["dataset"], value_vars=["test_acc", "f1_macro", "auc_macro"],
                  var_name="metric", value_name="value")
    # nice display names
    name_map = {"test_acc": "Accuracy", "f1_macro": "F1-macro", "auc_macro": "AUC-macro"}
    long["metric"] = long["metric"].map(name_map)

    # If some AUCs are NaN (e.g., multi-class without macro AUC), keep them—they just won’t show bars.
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=long, x="dataset", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Score")
    ax.set_title("GIN — Best Model Performance per Dataset")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Metric")

    # Labels on bars
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

    fig.tight_layout()
    out = out_dir / "gin_summary_best_performance_bars.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("Saved:", out)

def main():
    csv_path = find_csv()
    print("Reading:", csv_path)
    out_dir = csv_path.parent
    df = load_summary(csv_path)
    plot_hparams_alt(df, out_dir)

    plot_runtime_memory_combo(df, out_dir)
    plot_best_performance_bars(df, out_dir)
    

if __name__ == "__main__":
    main()
