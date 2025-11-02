import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re, json


class Config:
    STYLE = 'seaborn-v0_8'
    PALETTE = "husl"
    DPI = 300
    FIGSIZE = (15, 15)
    COLOR_FIT = '#2E86AB'
    COLOR_EMBED = '#A23B72'
    COLOR_PEAK = '#FF6B6B'
    BASE = Path(__file__).parent


    OUT_DIR   = BASE / "embeddings_gin_version2"               
    CSV       = OUT_DIR / "metrics_summary.csv"        
    RUNS_ROOT = BASE / "embeddings_gin_version2"                 
    PLOTS_DIR = OUT_DIR / "plots_GIN"

    # Confusion matrices for best-overall are always produced.
    # Set to True if you ALSO want a CM per embedding dim (slower; requires per-run CM files)
    PER_DIM_CONFUSION = False

plt.style.use(Config.STYLE)
sns.set_palette(Config.PALETTE)
sns.set_context("talk")
sns.set_style("whitegrid")
Config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# def _fmt_hp_row(r) -> str:
#     # No batch size
#     return f"ep={int(r['epochs'])}, lr={float(r['lr']):g}, drop={float(r['dropout']):g}"

def _title_hp_map_for_best_per_dim(b: pd.DataFrame, max_chars: int = 180) -> str:
    pairs = [f"{int(r.dim)}→({_fmt_hp_row(r)})" for _, r in b.iterrows()]
    s = "; ".join(pairs)
    return s if len(s) <= max_chars else (s[:max_chars-3] + "...")

def _title_hp_best_overall(r: pd.Series) -> str:
    return f"dim={int(r['dim'])}, {_fmt_hp_row(r)}"

def best_overall_row(df: pd.DataFrame, dataset: str) -> pd.Series | None:
    d = df[df["dataset"] == dataset].copy()
    if d.empty:
        return None
    if "val_acc" in d.columns and d["val_acc"].notna().any():
        d = d.sort_values(["val_acc", "test_acc", "total_time_s"], ascending=[False, False, True])
    else:
        d = d.sort_values(["test_acc", "total_time_s"], ascending=[False, True])
    return d.iloc[0]


def _fmt_hp_row(r) -> str:
    return f"dim={int(r['dim'])}, ep={int(r['epochs'])}, lr={float(r['lr']):g}, drop={float(r['dropout']):g}"


import re, json
_RUN_RE = re.compile(r"^gin_dim(?P<dim>\d+)_ep(?P<ep>\d+)_lr(?P<lr>[^_]+)_drop(?P<drop>[^_]+)$")

def _f(x):
    try:
        return float(x)
    except Exception:
        return None

def _match_by_folder_name(ds_dir: Path, r: pd.Series) -> Path | None:
    want_dim = int(r["dim"]); want_ep = int(r["epochs"])
    want_lr  = float(r["lr"]); want_dr = float(r["dropout"])
    # exact numeric match
    for p in ds_dir.iterdir():
        if not p.is_dir(): continue
        m = _RUN_RE.match(p.name)
        if not m: continue
        dim_i = int(m.group("dim")); ep_i = int(m.group("ep"))
        lr_i  = _f(m.group("lr"));   dr_i = _f(m.group("drop"))
        if lr_i is None or dr_i is None: continue
        if dim_i == want_dim and ep_i == want_ep and abs(lr_i - want_lr) < 1e-12 and abs(dr_i - want_dr) < 1e-12:
            return p
    # relaxed: same dim/ep; closest lr+drop
    candidate, bestd = None, 1e9
    for p in ds_dir.iterdir():
        if not p.is_dir(): continue
        m = _RUN_RE.match(p.name)
        if not m: continue
        dim_i = int(m.group("dim")); ep_i = int(m.group("ep"))
        lr_i  = _f(m.group("lr"));   dr_i = _f(m.group("drop"))
        if lr_i is None or dr_i is None: continue
        if dim_i == want_dim and ep_i == want_ep:
            d = abs(lr_i - want_lr) + abs(dr_i - want_dr)
            if d < bestd:
                bestd, candidate = d, p
    return candidate

def _match_by_metrics_json(ds_dir: Path, r: pd.Series) -> Path | None:
    want = {
        "embedding_dim": int(r["dim"]),
        "epochs": int(r["epochs"]),
        "lr": float(r["lr"]),
        "dropout": float(r["dropout"]),
    }
    best, best_score = None, -1
    for p in ds_dir.iterdir():
        if not p.is_dir(): continue
        mj = p / "metrics.json"
        if not mj.exists(): continue
        try:
            meta = json.loads(mj.read_text())
        except Exception:
            continue
        score = 0
        # give points for exact matches; prefer perfect match
        if int(meta.get("embedding_dim", -999)) == want["embedding_dim"]: score += 2
        if int(meta.get("epochs", -999)) == want["epochs"]: score += 2
        if abs(float(meta.get("lr", 1e9)) - want["lr"]) < 1e-12: score += 2
        if abs(float(meta.get("dropout", 1e9)) - want["dropout"]) < 1e-12: score += 2
        if score > best_score:
            best_score, best = score, p
            if score == 8:  # perfect match
                return p
    return best

def find_best_run_dir_for_row(r: pd.Series, runs_root: Path) -> Path | None:
    ds_dir = runs_root / str(r["dataset"])
    if not ds_dir.exists():
        return None
    # 1) folder-name match
    p = _match_by_folder_name(ds_dir, r)
    if p is not None:
        return p
    # 2) metrics.json match
    p = _match_by_metrics_json(ds_dir, r)
    return p


BASE_REQUIRED_SOFT = {
    "fit_time_s","embed_time_s","total_time_s",
    "test_acc","test_f1_macro","test_auc_macro",
    "epochs","lr","dropout","rss_before_mb","rss_after_mb","dim",
}

def load_prepare(path: Path) -> pd.DataFrame:
    if not path.exists():
        # try common fallback
        alt = path.parent / "metrics_summary.csv"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"CSV not found at {path} (or {alt})")

    df = pd.read_csv(path)

    # Header normalization
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["dataset", "name", "data", "graph_dataset"]:
            rename_map[c] = "dataset"
        elif cl in ["peak_tracemallov_mb", "peak_tracemalloc", "peak_tracemalloc_mb"]:
            rename_map[c] = "peak_tracemalloc_mb"
        elif cl in ["dim","embedding_dim"]:
            rename_map[c] = "dim"
        elif cl in ["it_time_s","train_time_sec"]:
            rename_map[c] = "fit_time_s"
        elif cl in ["embed_time_s","embed_time_sec"]:
            rename_map[c] = "embed_time_s"
        elif cl in ["total_time_s","total_time_sec"]:
            rename_map[c] = "total_time_s"
    df = df.rename(columns=rename_map)

    if "dataset" not in df.columns:
        raise ValueError(f" Could not find a dataset column in {path.name}. Found: {list(df.columns)}")

    if "peak_tracemalloc_mb" not in df.columns:
        print(" 'peak_tracemalloc_mb' not found — creating zeros.")
        df["peak_tracemalloc_mb"] = 0.0
    if "n_graphs" not in df.columns:
        df["n_graphs"] = np.nan
    if "embed_time_s" not in df.columns:
        df["embed_time_s"] = 0.0
    if "fit_time_s" not in df.columns:
        df["fit_time_s"] = 0.0

    missing = BASE_REQUIRED_SOFT - set(df.columns)
    if missing:
        raise ValueError(f" Missing required columns in CSV: {missing}")

    df["fit_embed_ratio"] = (df["fit_time_s"] / df["embed_time_s"]).replace([np.inf, -np.inf], np.nan)
    df["embed_percentage"] = (df["embed_time_s"] / df["total_time_s"]).replace([np.inf,-np.inf], np.nan) * 100
    df["dataset"] = df["dataset"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    df["dataset"] = df["dataset"].str.replace("EMDB-MULTI", "IMDB-MULTI", regex=False)

    # numeric coercion
    for c in ["dim","epochs","lr","dropout","fit_time_s","embed_time_s","total_time_s",
              "test_acc","test_f1_macro","test_auc_macro","val_acc","peak_tracemalloc_mb",
              "rss_before_mb","rss_after_mb","batch_size","n_graphs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def best_per_dim(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    d = df[df["dataset"] == dataset].dropna(subset=[metric]).copy()
    if d.empty:
        return d
    idx = d.groupby("dim")[metric].idxmax()
    return d.loc[idx].sort_values("dim")

def best_overall(df: pd.DataFrame, dataset: str, metric: str) -> pd.Series | None:
    d = df[df["dataset"] == dataset].dropna(subset=[metric]).copy()
    if d.empty:
        return None
    # tie-break by higher test_acc, then lower total_time
    d = d.sort_values([metric,"test_acc","total_time_s"], ascending=[False,False,True])
    return d.iloc[0]


_RUN_RE = re.compile(r"^gin_dim(?P<dim>\d+)_ep(?P<ep>\d+)_lr(?P<lr>[^_]+)_drop(?P<drop>[^_]+)$")
def _f(x):
    try: return float(x)
    except: return None

def find_run_dir(row: pd.Series) -> Path | None:
    ds_dir = Config.RUNS_ROOT / str(row["dataset"])
    if not ds_dir.exists():
        return None
    want_dim = int(row["dim"])
    want_ep  = int(row["epochs"])
    want_lr  = float(row["lr"])
    want_drop= float(row["dropout"])

    # exact numeric
    for p in ds_dir.iterdir():
        if not p.is_dir(): continue
        m = _RUN_RE.match(p.name)
        if not m: continue
        dim_i, ep_i = int(m.group("dim")), int(m.group("ep"))
        lr_i, drop_i = _f(m.group("lr")), _f(m.group("drop"))
        if lr_i is None or drop_i is None: continue
        if dim_i==want_dim and ep_i==want_ep and abs(lr_i-want_lr)<1e-12 and abs(drop_i-want_drop)<1e-12:
            return p

    # relaxed: same dim/ep, closest lr+drop
    best, bd = None, 1e9
    for p in ds_dir.iterdir():
        if not p.is_dir(): continue
        m = _RUN_RE.match(p.name)
        if not m: continue
        dim_i, ep_i = int(m.group("dim")), int(m.group("ep"))
        lr_i, drop_i = _f(m.group("lr")), _f(m.group("drop"))
        if lr_i is None or drop_i is None: continue
        if dim_i==want_dim and ep_i==want_ep:
            d = abs(lr_i-want_lr)+abs(drop_i-want_drop)
            if d < bd:
                bd, best = d, p
    return best

def line_vs_dim(df: pd.DataFrame, metric: str, label: str, fname: str):
    fig = plt.figure(figsize=Config.FIGSIZE)

    subtitle_lines = []
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, metric)
        if b.empty:
            continue
        plt.plot(b["dim"], b[metric], marker="o", linewidth=2, label=ds)
        subtitle_lines.append(f"{ds}: " + _title_hp_map_for_best_per_dim(b))

    plt.xlabel("Embedding Dimension")
    plt.ylabel(label)
    plt.title(f"GIN — {label} vs Embedding Dimension", pad=12)

    # put the hp map as a smaller subtitle
    if subtitle_lines:
        plt.suptitle("\n".join(subtitle_lines), y=0.99, fontsize=11)

    plt.grid(True, alpha=0.3); plt.legend()
    out = Config.PLOTS_DIR / fname
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out, dpi=Config.DPI); plt.close()
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
        hp = _title_hp_map_for_best_per_dim(b)
        plt.title(f"{ds} — GIN Time Breakdown\n{hp}")
        plt.legend(); plt.grid(True, which="both", alpha=0.3)
        out = Config.PLOTS_DIR / f"gin_time_breakdown_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

def memory_plots(df: pd.DataFrame):
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        x = np.arange(len(b)); w = 0.25
        fig = plt.figure(figsize=Config.FIGSIZE)
        plt.bar(x - w, b["rss_before_mb"], width=w, label="Before", color="#4ECDC4")
        plt.bar(x,      b["rss_after_mb"], width=w, label="After",  color="#2E86AB")
        plt.bar(x + w,  b["peak_tracemalloc_mb"], width=w, label="Peak Tracemalloc", color=Config.COLOR_PEAK)
        plt.xticks(x, b["dim"].astype(int))
        plt.xlabel("Embedding Dimension"); plt.ylabel("Memory (MB)")
        hp = _title_hp_map_for_best_per_dim(b)
        plt.title(f"{ds} — Memory Usage vs Dimension\n{hp}")
        plt.legend(framealpha=0.9); plt.grid(True, axis="y", alpha=0.3)
        out = Config.PLOTS_DIR / f"gin_memory_bars_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

    fig = plt.figure(figsize=Config.FIGSIZE)
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty: continue
        plt.bar(b["dim"] + np.random.uniform(-2, 2, size=len(b)),
                b["peak_tracemalloc_mb"], width=10, alpha=0.7, label=ds)
    plt.xlabel("Embedding Dimension"); plt.ylabel("Peak Tracemalloc (MB)")
    plt.title("GIN — Peak Memory Usage (Bar Plot per Dataset)")
    plt.grid(True, alpha=0.3); plt.legend()
    out = Config.PLOTS_DIR / "gin_peak_tracemalloc_barplot.png"
    plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
    print("Saved:", out)

def tradeoff_dual_axis(df: pd.DataFrame):
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        dims = b["dim"].astype(int).values
        acc  = b["test_acc"].values
        tsec = b["total_time_s"].values
        x = np.arange(len(dims))

        fig, ax1 = plt.subplots(figsize=Config.FIGSIZE)
        ax1.bar(x, acc, color="#4ECDC4", alpha=0.85, label="Accuracy")
        ax1.set_xlabel("Embedding Dimension"); ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, max(1.0, np.nanmax(acc)*1.05))
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

        hp = _title_hp_map_for_best_per_dim(b)
        plt.title(f"{ds} — Accuracy vs Compute Cost\n{hp}")
        out = Config.PLOTS_DIR / f"gin_tradeoff_dualaxis_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

def efficiency_plot(df: pd.DataFrame):
    for ds in sorted(df["dataset"].unique()):
        b = best_per_dim(df, ds, "test_acc")
        if b.empty:
            continue
        dims = b["dim"].astype(int).values
        eff = (b["test_acc"] / b["total_time_s"]).replace([np.inf, -np.inf], np.nan).values
        fig = plt.figure(figsize=Config.FIGSIZE)
        plt.plot(dims, eff, marker="o", linewidth=2, color="#45B7D1")
        plt.xlabel("Embedding Dimension"); plt.ylabel("Accuracy per Second")
        hp = _title_hp_map_for_best_per_dim(b)
        plt.title(f"{ds} — Efficiency (Accuracy/Second)\n{hp}")
        plt.grid(True, alpha=0.3)
        out = Config.PLOTS_DIR / f"gin_efficiency_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)

def tradeoff_heatmap(df: pd.DataFrame):
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
        hp = _title_hp_map_for_best_per_dim(b)
        plt.title(f"{ds} — Tradeoff Heatmap\n{hp}")
        plt.xlabel("Metric"); plt.ylabel("Embedding Dimension")
        out = Config.PLOTS_DIR / f"gin_tradeoff_heatmap_{ds}.png"
        plt.tight_layout(); plt.savefig(out, dpi=Config.DPI); plt.close()
        print("Saved:", out)


def epochs_lr_heatmaps_subplots(df: pd.DataFrame):
    datasets = sorted(df["dataset"].unique())
    lrs = sorted(df["lr"].dropna().unique())
    epochs_list = sorted(df["epochs"].dropna().unique())
    if len(lrs) == 0 or len(epochs_list) == 0:
        print("No lr/epochs grid to plot.")
        return

    n = len(datasets)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 15), squeeze=False)

    for idx, ds in enumerate(datasets):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub = df[df["dataset"] == ds]

        grid = np.full((len(epochs_list), len(lrs)), np.nan)
        for i, ep in enumerate(epochs_list):
            for j, lr in enumerate(lrs):
                cand = sub[(sub["epochs"] == ep) & (sub["lr"] == lr)]
                if not cand.empty:
                    grid[i, j] = cand["test_acc"].max()

        sns.heatmap(
            grid, annot=True, fmt=".3f", cmap="YlOrRd",
            xticklabels=[f"{x:g}" for x in lrs],
            yticklabels=[int(e) for e in epochs_list],
            ax=ax, cbar=(idx == 0)
        )
        ax.set_xlabel("lr"); ax.set_ylabel("epochs")

        if not sub.empty and "test_acc" in sub.columns:
            best_row = sub.loc[sub["test_acc"].idxmax()]
            best_dim = int(best_row["dim"])
            # best_bs  = int(best_row["batch_size"]) if pd.notna(best_row["batch_size"]) else "?"
            best_ep  = int(best_row["epochs"])
            best_lr  = float(best_row["lr"])
            best_acc = float(best_row["test_acc"])
            # ax.set_title(f"{ds}\nBest: dim={best_dim}, bs={best_bs}, ep={best_ep}, lr={best_lr:g}, acc={best_acc:.3f}",
            #              fontsize=15)
            ax.set_title(
                f"{ds}\nBest: dim={best_dim}, ep={best_ep}, lr={best_lr:g}, acc={best_acc:.3f}",
                fontsize=15
            )
            if best_ep in epochs_list and best_lr in lrs:
                i = epochs_list.index(best_ep); j = lrs.index(best_lr)
                ax.scatter(j + 0.5, i + 0.5, marker="*", s=250, color="white",
                           edgecolor="black", linewidths=0.8, zorder=3)
        else:
            ax.set_title(f"{ds}", fontsize=12)

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



def falsifications_from_cm(cm: np.ndarray) -> pd.DataFrame:
    cm = np.asarray(cm)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    fn = support - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)
    mis = np.divide(fn, support, out=np.zeros_like(fn, dtype=float), where=support>0)
    return pd.DataFrame({
        "class": np.arange(cm.shape[0]),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Support": support, "MisclassificationRate": mis
    })

def plot_confusion_matrix(cm, title, out_path, labels=None):
    plt.figure(figsize=Config.FIGSIZE)
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                     xticklabels=labels if labels is not None else "auto",
                     yticklabels=labels if labels is not None else "auto")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=Config.DPI); plt.close()
    print("Saved:", out_path)

def plot_fp_fn(df_fals, title_prefix, out_dir: Path):
    out1 = out_dir / f"{title_prefix}_FP_FN.png"
    x = np.arange(len(df_fals))
    plt.figure(figsize=(12,6))
    plt.bar(x - 0.2, df_fals["FP"], width=0.4, color="#FF6B6B", label="False Positives")   # red/pink
    plt.bar(x + 0.2, df_fals["FN"], width=0.4, color="#4ECDC4", label="False Negatives")   # teal

    plt.xticks(x, df_fals["class"].astype(str))
    plt.xlabel("Class"); plt.ylabel("Count")
    plt.title(f"{title_prefix} — False Positives / False Negatives")
    plt.legend(); plt.tight_layout(); plt.savefig(out1, dpi=Config.DPI); plt.close()
    print("Saved:", out1)

    out2 = out_dir / f"{title_prefix}_misclassification.png"
    plt.figure(figsize=(12,6))
    sns.barplot(x="class", y="MisclassificationRate", data=df_fals)
    plt.ylim(0, 1.0)
    plt.xlabel("Class"); plt.ylabel("Misclassification Rate")
    plt.title(f"{title_prefix} — Misclassification Rate")
    plt.tight_layout(); plt.savefig(out2, dpi=Config.DPI); plt.close()
    print("Saved:", out2)

# def confusion_and_falsifications(df: pd.DataFrame):
#     # Best overall per dataset (by val_acc if present else test_acc)
#     rows = []
#     metric = "val_acc" if "val_acc" in df.columns and df["val_acc"].notna().any() else "test_acc"
#     for ds in sorted(df["dataset"].unique()):
#         r = best_overall(df, ds, metric)
#         if r is None:
#             continue
#         rows.append(r)
#     if rows:
#         pd.DataFrame(rows).to_csv(Config.PLOTS_DIR / "best_overall_selection.csv", index=False)

#     # Produce CM & falsifications for best overall
#     for r in rows:
#         ds = str(r["dataset"])
#         run_dir = find_run_dir(r)
#         if run_dir is None:
#             print(f" {ds}: no matching run dir found under {Config.RUNS_ROOT/ds}")
#             continue
#         cm_path = run_dir / "test_confusion_matrix.csv"
#         if not cm_path.exists():
#             print(f"{ds}: missing {cm_path}")
#             continue
#         cm = np.loadtxt(cm_path, delimiter=",").astype(int)
#         labels = list(range(cm.shape[0]))
#         prefix = f"{ds}_GIN_best"

#         hp_overall = _title_hp_best_overall(r)
#         plot_confusion_matrix(
#             cm,
#             f"{ds} — GIN — Confusion Matrix (Best Overall)\n{hp}",
#             Config.PLOTS_DIR / f"{prefix}_cm.png",
#             labels=labels
#         )


#         fals = falsifications_from_cm(cm)
#         fals.to_csv(Config.PLOTS_DIR / f"{prefix}_falsifications.csv", index=False)
#         hp = _title_hp_best_overall(r)  # from the same row used for that dataset
#         plot_fp_fn(df_fals, f"{ds} — GIN (Best Overall)\n{hp}", Config.PLOTS_DIR)

def confusion_and_falsifications(df: pd.DataFrame):
    """
    Ensures CM / FP-FN / misclassification are computed from the **best overall**
    hyperparameters per dataset. Titles include the hyperparameters.
    """
    rows = []
    for ds in sorted(df["dataset"].unique()):
        r = best_overall_row(df, ds)
        if r is None:
            print(f"[Skip] No rows for {ds}")
            continue
        rows.append(r)

    # Save the selection snapshot
    if rows:
        pd.DataFrame(rows).to_csv(Config.PLOTS_DIR / "best_overall_selection.csv", index=False)

    # For each dataset, load CM from the exact best run folder
    for r in rows:
        ds = str(r["dataset"])
        hp_str = _fmt_hp_row(r)
        run_dir = find_best_run_dir_for_row(r, Config.RUNS_ROOT)
        if run_dir is None:
            print(f"{ds}: could not locate run folder for best hyperparameters ({hp_str})")
            continue

        cm_path = run_dir / "test_confusion_matrix.csv"
        if not cm_path.exists():
            print(f" {ds}: missing {cm_path} (best hyperparameters: {hp_str})")
            continue

        # Load CM and compute per-class stats
        cm = np.loadtxt(cm_path, delimiter=",").astype(int)
        labels = list(range(cm.shape[0]))
        prefix = f"{ds}_GIN_best"

        # Confusion matrix with HPs in title
        plot_confusion_matrix(
            cm,
            f"{ds} — GIN — Confusion Matrix (Best Overall)\n{hp_str}",
            Config.PLOTS_DIR / f"{prefix}_cm.png",
            labels=labels
        )

        # FP/FN/misclassification from that CM
        fals = falsifications_from_cm(cm)
        fals.to_csv(Config.PLOTS_DIR / f"{prefix}_falsifications.csv", index=False)
        plot_fp_fn(fals, f"{ds} — GIN (Best Overall)\n{hp_str}", Config.PLOTS_DIR)

    # Optionally: CM per embedding dim (if available)
    if Config.PER_DIM_CONFUSION:
        print("Info: PER_DIM_CONFUSION is enabled — generating per-dim confusion matrices...")
        for ds in sorted(df["dataset"].unique()):
            b = best_per_dim(df, ds, metric)
            for _, r in b.iterrows():
                run_dir = find_run_dir(r)
                if run_dir is None:
                    continue
                cm_path = run_dir / "test_confusion_matrix.csv"
                if not cm_path.exists():
                    continue
                cm = np.loadtxt(cm_path, delimiter=",").astype(int)
                labels = list(range(cm.shape[0]))
                prefix = f"{ds}_GIN_dim{int(r['dim'])}"
                plot_confusion_matrix(cm, f"{ds} — GIN — Confusion Matrix (dim={int(r['dim'])})",
                                      Config.PLOTS_DIR / f"{prefix}_cm.png", labels=labels)
                fals = falsifications_from_cm(cm)
                fals.to_csv(Config.PLOTS_DIR / f"{prefix}_falsifications.csv", index=False)
                plot_fp_fn(fals, f"{ds} — GIN (dim={int(r['dim'])})", Config.PLOTS_DIR)

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
            "lr": r["lr"],
            "val_acc": (r["val_acc"] if "val_acc" in r else np.nan),
            "test_acc": r["test_acc"],
            "f1_macro": r["test_f1_macro"],
            "auc_macro": r["test_auc_macro"],
            "total_time_s": r["total_time_s"],
            "fit_time_s": r["fit_time_s"],
            "embed_time_s": r["embed_time_s"],
            "rss_before_mb": r["rss_before_mb"],
            "rss_after_mb": r["rss_after_mb"],
            "peak_tracemalloc_mb": r.get("peak_tracemalloc_mb", np.nan),
        })
    tbl = pd.DataFrame(rows).sort_values("dataset")
    out = Config.PLOTS_DIR / "best_gin_summary.csv"
    tbl.to_csv(out, index=False)
    print("Saved summary CSV ->", out)
    if not tbl.empty:
        print(tbl.to_string(index=False))

def main():
    print("Reading:", Config.CSV)
    print("Runs root:", Config.RUNS_ROOT)
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

    # Hyperparam surface
    epochs_lr_heatmaps_subplots(df)

    # Confusion matrices + falsifications
    confusion_and_falsifications(df)

    # Best config summary
    summary_csv(df)

    print(f"\n All plots & reports saved to: {Config.PLOTS_DIR}")

if __name__ == "__main__":
    main()
