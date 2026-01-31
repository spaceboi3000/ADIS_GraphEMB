from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 




class Config:
    # OUT_DIR = Path("../../plots/plots_classification_before_permutation/")
    # GIN_PATH = Path("../../results/gin_classification_results.csv")
    # G2V_CLS_PATH = Path("../../results/graph2vec_classification_results.csv")
    # NETLSD_CLS_PATH = Path("../../results/netlsd_classification_results.csv")

    # G2V_METRICS_PATH = Path("../../embeddings/embeddings_graph2vec/metrics_graph2vec.csv")
    # NETLSD_METRICS_PATH = Path("../../embeddings/embeddings_netlsd/metrics_netlsd.csv")
    # METRICS_SUMMARY_PATH = Path("../../embeddings/embeddings_gin/metrics_summary.csv")
    
    #paths need change for before and after permutation
    
    OUT_DIR = Path("../../plots/plots_classificarion_after_permutation/")
    
    GIN_PATH = Path("../../results/perturbated_gin_classification_results.csv")
    G2V_CLS_PATH = Path("../../results/perturbated_graph2vec_classification_results.csv")
    NETLSD_CLS_PATH = Path("../../results/perturbated_netlsd_classification_results.csv")
    
    G2V_METRICS_PATH = Path("../../permutated_embeddings/permutated_graph2vec/permutated_metrics.csv")
    NETLSD_METRICS_PATH = Path("../../permutated_embeddings/permutated_netlsd/permutated_metrics.csv")
    METRICS_SUMMARY_PATH = Path("../../permutated_embeddings/permutated_gin/metrics_summary.csv")
    

    MAKE_PER_DATASET = True
    MAKE_AVG_PLOTS = True

    USE_MAX_DIM_PER_METHOD = True
    USE_FIXED_DIM = False
    FIXED_DIM = 128

    FIG_SIZE = (10, 6)
    FIG_SIZE_WIDE = (14, 6)
    DPI = 300
    FONT_SIZE = 11

    # white bg
    FIG_FACE = "white"
    AX_FACE = "white"
    SAVE_FACE = "white"

    @classmethod
    def setup(cls):
        cls.OUT_DIR.mkdir(parents=True, exist_ok=True)

        # plt.style.use("default")
        plt.style.use("bmh")
        plt.rcParams.update({  "font.size": cls.FONT_SIZE,"figure.figsize": cls.FIG_SIZE, "axes.labelsize": cls.FONT_SIZE, "axes.titlesize": cls.FONT_SIZE + 1, "xtick.labelsize": cls.FONT_SIZE - 1, "ytick.labelsize": cls.FONT_SIZE - 1, "legend.fontsize": cls.FONT_SIZE - 1, "figure.facecolor": cls.FIG_FACE, "axes.facecolor": cls.AX_FACE, "savefig.facecolor": cls.SAVE_FACE, "savefig.transparent": False,})


# DATA PROCESSING
class DataProcessor:
    
    @staticmethod
    def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_csv(path) if path.exists() else None

    @staticmethod
    def parse_dimension(x) -> float:
        
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, np.integer)):
            return int(x)
        
        s = str(x).strip()
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else np.nan

    @staticmethod
    def standardize_method_name(s: str) -> str:
      
        if s is None:
            return "Unknown"
        s = str(s).strip()
        low = s.lower()
       
        if low in ["gin", "graph isomorphism network"]:
            return "GIN"
        if low.startswith("graph2vec"):
            return "Graph2Vec"
        if low.startswith("netlsd"):
            return "NetLSD"
        return s
    
    
    @classmethod
    def process_gin_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        df["method"] = "GIN"
        df["dataset"] = df["dataset"].astype(str)
        df["dim"] = df["dim"].apply(cls.parse_dimension)

        # Metrics
        df["accuracy"] = df.get("test_acc", np.nan)
        df["f1"] = df.get("test_f1_macro", np.nan)
        df["auc"] = df.get("test_auc_macro", np.nan)

        # Timing
        df["clf_train_time_s"] = df.get("total_time_s", np.nan)
        df["embed_time_s"] = df.get("embed_time_s", np.nan)
        df["fit_time_s"] = np.nan

        # Memory
        df["peak_tracemalloc_mb"] = df.get("peak_tracemalloc_mb", np.nan)
        df["rss_before_mb"] = df.get("rss_before_mb", np.nan)
        df["rss_after_mb"] = df.get("rss_after_mb", np.nan)
        df["rss_delta_mb"] = df["rss_after_mb"] - df["rss_before_mb"]

        return df[cls._get_standard_columns()]


    @classmethod
    def process_unsupervised_data( cls, cls_df: pd.DataFrame, metrics_df: Optional[pd.DataFrame], method_name: str) -> pd.DataFrame:
        
        df = cls_df.copy()
        df["method"] = df.get("Method", method_name).apply(cls.standardize_method_name)
        df["dataset"] = df.get("Dataset", df.get("dataset")).astype(str)
        df["dim"] = df.get("Dim", df.get("dim")).apply(cls.parse_dimension)

        # Metrics
        df["accuracy"] = df.get("Accuracy", df.get("test_acc", np.nan))
        df["f1"] = df.get("F1", df.get("test_f1_macro", np.nan))
        df["auc"] = df.get("AUC", df.get("test_auc_macro", np.nan))
        df["clf_train_time_s"] = df.get("TrainTime", df.get("clf_train_time_s", np.nan))

        # Initialize timing/memory columns
        for c in ["embed_time_s", "fit_time_s", "peak_tracemalloc_mb", "rss_before_mb", "rss_after_mb"]:
            if c not in df.columns:
                df[c] = np.nan
        df["rss_delta_mb"] = np.nan

        if metrics_df is not None and not metrics_df.empty:
            df = cls._merge_metrics(df, metrics_df, method_name)

        if df["rss_before_mb"].notna().any() and df["rss_after_mb"].notna().any():
            df["rss_delta_mb"] = df["rss_after_mb"] - df["rss_before_mb"]

        for c in cls._get_standard_columns():
            if c not in df.columns:
                df[c] = np.nan

        return df[cls._get_standard_columns()]

    @classmethod
    def _merge_metrics(cls, df: pd.DataFrame, metrics_df: pd.DataFrame, method_name: str) -> pd.DataFrame:
       
        m = metrics_df.copy()
        m["method"] = m.get("method", method_name).apply(cls.standardize_method_name)
        m["dataset"] = m.get("dataset", m.get("Dataset")).astype(str)
        m["dim"] = m.get("dim", m.get("Dim")).apply(cls.parse_dimension)

        merge_cols = ["method", "dataset", "dim"]
        possible_cols = ["fit_time_s", "embed_time_s", "pca_time_s",
                         "rss_before_mb", "rss_after_mb", "peak_tracemalloc_mb"]
        keep_cols = [c for c in possible_cols if c in m.columns]

        if keep_cols:
            mm = m[merge_cols + keep_cols].drop_duplicates()
            df = df.merge(mm, on=merge_cols, how="left", suffixes=("", "_m"))

            for c in keep_cols:
                if f"{c}_m" in df.columns:
                    df[c] = df[c].combine_first(df[f"{c}_m"])
                    df.drop(columns=[f"{c}_m"], inplace=True)

            if "pca_time_s" in df.columns and "embed_time_s" in df.columns:
                df["embed_time_s"] = df["embed_time_s"].combine_first(df["pca_time_s"])
            elif "pca_time_s" in df.columns:
                df["embed_time_s"] = df["pca_time_s"]

        return df

    @staticmethod
    def _get_standard_columns() -> list:
        return [ "method", "dataset", "dim", "accuracy", "f1", "auc","clf_train_time_s", "embed_time_s", "fit_time_s", "peak_tracemalloc_mb", "rss_before_mb", "rss_after_mb", "rss_delta_mb",  ]


# PLOTTING
class Plotter:
    
    @staticmethod
    def _force_white(fig, ax=None):
        
        fig.patch.set_facecolor(Config.FIG_FACE)
        if ax is None:
            for a in fig.get_axes():
                a.set_facecolor(Config.AX_FACE)
        else:
            ax.set_facecolor(Config.AX_FACE)


    @staticmethod
    def save_figure(fig, filename: str):
        
        Plotter._force_white(fig)
        out_path = Config.OUT_DIR / filename
        fig.tight_layout()
        fig.savefig(out_path, dpi=Config.DPI, bbox_inches="tight", facecolor=Config.SAVE_FACE)
        plt.close(fig)
        print(f"  Saved: {filename}")

    @classmethod
    def grouped_metric_bars(cls, df: pd.DataFrame, title: str, filename: str):
        
        plot_df = df.copy()
        for c in ["accuracy", "f1", "auc"]:
            if c not in plot_df.columns:
                plot_df[c] = np.nan
        plot_df = plot_df.dropna(subset=["accuracy", "f1", "auc"], how="all")
        if plot_df.empty:
            return

        if plot_df["accuracy"].notna().any():
            plot_df = plot_df.sort_values("accuracy", ascending=False)

        methods = plot_df["method"].tolist()
        x = np.arange(len(methods))
        width = 0.25
        
        
        fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
        ax.bar(x - width, plot_df["accuracy"], width, label="Accuracy", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.bar(x, plot_df["f1"], width, label="F1-Score", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.bar(x + width, plot_df["auc"], width, label="AUC", alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.25)
        cls.save_figure(fig, filename)


    @classmethod
    def metric_vs_dimension_lines(cls, df: pd.DataFrame, metric_col: str, title: str, filename: str):
        plot_df = df.dropna(subset=["dim", metric_col]).copy()
        if plot_df.empty:
            return

        fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
        for method, g in plot_df.groupby("method"):
            g = g.sort_values("dim")
            ax.plot(g["dim"], g[metric_col], marker="o", label=method, linewidth=2, markersize=6)
        
        
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel(metric_col.replace("_", " ").title())
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

        cls.save_figure(fig, filename)
        
        
    @classmethod
    def accuracy_efficiency_scatter(cls, df: pd.DataFrame, time_col: str, title: str, filename: str):
        
        plot_df = df.dropna(subset=[time_col, "accuracy"]).copy()
        if plot_df.empty:
            return

        fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
        for method, method_df in plot_df.groupby("method"):
            ax.scatter(  method_df[time_col], method_df["accuracy"], label=method, alpha=0.75, s=100, edgecolors="black", linewidth=0.4)
            
            for _, row in method_df.iterrows():
                dim_val = row.get("dim", np.nan)
                if pd.notna(dim_val):
                    ax.annotate(  f"{int(dim_val)}",(row[time_col], row["accuracy"]), xytext=(5, 5), textcoords="offset points", fontsize=8, alpha=0.8 )

        ax.set_xlabel(time_col.replace("_", " ").title())
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        cls.save_figure(fig, filename)


    @classmethod
    def performance_heatmap(cls, df: pd.DataFrame, metric_col: str, title: str, filename: str):
        plot_df = df.dropna(subset=["method", "dim", metric_col]).copy()
        if plot_df.empty:
            return

        pivot = plot_df.pivot_table(index="method", columns="dim", values=metric_col, aggfunc="mean")
        if pivot.empty:
            return

        fig, ax = plt.subplots(figsize=Config.FIG_SIZE_WIDE)
        cls._force_white(fig, ax)
        sns.heatmap( pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": metric_col.replace("_", " ").title()},  ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Method")
        cls.save_figure(fig, filename)


    @classmethod
    def memory_usage_bars(cls, df: pd.DataFrame, title: str, filename: str):
        plot_df = df.dropna(subset=["peak_tracemalloc_mb"]).copy()
        if plot_df.empty:
            return

        plot_df = plot_df.sort_values("peak_tracemalloc_mb", ascending=False)
        fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
        ax.bar(range(len(plot_df)), plot_df["peak_tracemalloc_mb"], alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(plot_df)))
        
        ax.set_xticklabels( [f"{m}\n(d={int(d)})" for m, d in zip(plot_df["method"], plot_df["dim"])],  rotation=45, ha="right")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        cls.save_figure(fig, filename)



    @classmethod
    def accuracy_vs_compute_cost_dual(cls, df: pd.DataFrame, title: str, filename: str):
        
        plot_df = df.dropna(subset=["dim", "accuracy"]).copy()
        if plot_df.empty:
            return

        plot_df["total_time"] = plot_df["clf_train_time_s"].fillna(0) + plot_df["embed_time_s"].fillna(0)
        plot_df = plot_df.dropna(subset=["total_time"])
        if plot_df.empty:
            return

        fig, ax1 = plt.subplots(figsize=Config.FIG_SIZE_WIDE)
        ax1.set_xlabel("Embedding Dimension")
        ax1.set_ylabel("Accuracy")

        for method, g in plot_df.groupby("method"):
            g = g.sort_values("dim")
            ax1.plot(g["dim"], g["accuracy"], marker="o", label=f"{method} (Acc)", linewidth=2, markersize=6, alpha=0.9)

        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Total Compute Time (s)")
        for method, g in plot_df.groupby("method"):
            g = g.sort_values("dim")
            ax2.plot(g["dim"], g["total_time"], marker="s", label=f"{method} (Time)", linewidth=2, markersize=6, linestyle="--", alpha=0.9)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)
        fig.suptitle(title)
        cls.save_figure(fig, filename)
        
        
    @classmethod
    def accuracy_vs_compute_cost_pareto(cls, df: pd.DataFrame, title: str, filename: str):
        plot_df = df.copy()
        plot_df["total_time"] = plot_df["clf_train_time_s"].fillna(0) + plot_df["embed_time_s"].fillna(0)
        plot_df["total_memory"] = plot_df["peak_tracemalloc_mb"].fillna(0)

        time_max = plot_df["total_time"].max()
        mem_max = plot_df["total_memory"].max()

        plot_df["norm_time"] = (plot_df["total_time"] / time_max) if time_max > 0 else 0
        plot_df["norm_memory"] = (plot_df["total_memory"] / mem_max) if mem_max > 0 else 0
        plot_df["compute_cost"] = plot_df["norm_time"] * 0.7 + plot_df["norm_memory"] * 0.3
        plot_df = plot_df.dropna(subset=["accuracy", "compute_cost"])
        
        if plot_df.empty:
            return

        fig, ax = plt.subplots(figsize=Config.FIG_SIZE_WIDE)

        for method, method_df in plot_df.groupby("method"):
            method_df = method_df.sort_values("dim")
            
            ax.plot(method_df["compute_cost"], method_df["accuracy"], alpha=0.25, linewidth=1)
            ax.scatter(  method_df["compute_cost"], method_df["accuracy"], s=100, alpha=0.75, edgecolors="black", linewidth=0.5, label=method )
           
            for _, row in method_df.iterrows():
                dim_val = row.get("dim", np.nan)
                if pd.notna(dim_val):
                    ax.annotate(
                        f"{int(dim_val)}",
                        (row["compute_cost"], row["accuracy"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, alpha=0.8
                    )

        # Pareto frontier
        all_points = plot_df[["compute_cost", "accuracy"]].values
        pareto_mask = np.ones(len(all_points), dtype=bool)
        for i, point in enumerate(all_points):
            dominated = np.any((all_points[:, 0] <= point[0]) & (all_points[:, 1] > point[1]))
            if dominated:
                pareto_mask[i] = False

        pareto_points = all_points[pareto_mask]
        if len(pareto_points) > 1:
            pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
            ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], "k--", alpha=0.35, linewidth=1.5, label="Pareto Frontier")

        ax.set_xlabel("Normalized Compute Cost (Time×0.7 + Memory×0.3)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
       
        ax.text(0.02, 0.98, "← Better\n(Lower Cost)",  transform=ax.transAxes, fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.25),)
        ax.text(  0.98, 0.02, "Better →\n(Higher Accuracy)",  transform=ax.transAxes, fontsize=9, ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.25),)
        cls.save_figure(fig, filename)



    @classmethod
    def accuracy_vs_compute_cost_faceted(cls, df: pd.DataFrame, title_prefix: str, filename: str):
        plot_df = df.dropna(subset=["dim", "accuracy"]).copy()
        if plot_df.empty:
            return

        cost_metrics = [ ("clf_train_time_s", "Training Time (s)"),  ("embed_time_s", "Embedding Time (s)"), ("peak_tracemalloc_mb", "Peak Memory (MB)"), ]
        
        available_metrics = [(col, label) for col, label in cost_metrics
                             if col in plot_df.columns and plot_df[col].notna().any()]
        if not available_metrics:
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, available_metrics):
            
            temp_df = plot_df.dropna(subset=[col])
            for method, g in temp_df.groupby("method"):
                g = g.sort_values("dim")
                sizes = (g["accuracy"] * 500).clip(50, 500)
                ax.scatter(g["dim"], g[col], s=sizes, alpha=0.65, label=method,
                           edgecolors="black", linewidth=0.5)
                ax.plot(g["dim"], g[col], alpha=0.25, linewidth=1)

            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs Dimension\n(bubble size = accuracy)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.25)

        fig.suptitle(title_prefix, fontsize=12, y=1.02)
        cls.save_figure(fig, filename)


    # STACKED BARS  
    @classmethod
    def stacked_time_memory_bars(cls, df: pd.DataFrame, title: str, filename: str):
        plot_df = df.copy()

        for c in ["clf_train_time_s", "embed_time_s", "fit_time_s","rss_before_mb", "rss_after_mb", "rss_delta_mb", "peak_tracemalloc_mb"]:
            if c in plot_df.columns:
                plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

        g = plot_df.groupby("method", as_index=False).mean(numeric_only=True)
        if g.empty:
            return

        g["embed_time_s"] = g.get("embed_time_s", np.nan).fillna(0)
        g["clf_train_time_s"] = g.get("clf_train_time_s", np.nan).fillna(0)
        g["fit_time_s"] = g.get("fit_time_s", np.nan).fillna(0)

        g["rss_before_mb"] = g.get("rss_before_mb", np.nan).fillna(0)
        g["rss_delta_mb"] = g.get("rss_delta_mb", np.nan).fillna(0)
        peak = g.get("peak_tracemalloc_mb", pd.Series([np.nan] * len(g))).copy()

        methods = g["method"].tolist()
        x = np.arange(len(methods))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # TIME STACKED
        bottom = np.zeros(len(g))
        ax1.bar(x, g["embed_time_s"], bottom=bottom, label="Embedding / Generation time (s)",
                alpha=0.9, edgecolor="black", linewidth=0.3)
        bottom += g["embed_time_s"].values

        ax1.bar(x, g["clf_train_time_s"], bottom=bottom, label="Classifier training time (s)",
                alpha=0.9, edgecolor="black", linewidth=0.3)
        bottom += g["clf_train_time_s"].values

        if (g["fit_time_s"] > 0).any():
            ax1.bar(x, g["fit_time_s"], bottom=bottom, label="Fit time (s)",
                    alpha=0.9, edgecolor="black", linewidth=0.3)

        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=25, ha="right")
        ax1.set_ylabel("Seconds")
        ax1.set_title("Time Breakdown (stacked)")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend(loc="best", fontsize=9)

        # MEMORY STACKED
        bottom = np.zeros(len(g))
        has_rss = (g["rss_before_mb"] > 0).any() or (g["rss_delta_mb"] > 0).any()
        if has_rss:
            ax2.bar(x, g["rss_before_mb"], bottom=bottom, label="RSS before (MB)",
                    alpha=0.9, edgecolor="black", linewidth=0.3)
            bottom += g["rss_before_mb"].values
            ax2.bar(x, g["rss_delta_mb"], bottom=bottom, label="RSS delta (MB)",
                    alpha=0.9, edgecolor="black", linewidth=0.3)

            if peak.notna().any():
                ax2.plot(x, peak.values, marker="o", linestyle="--", linewidth=1.5,
                         label="Peak tracemalloc (MB)", alpha=0.9)
        else:
            if peak.notna().any():
                ax2.bar(x, peak.values, label="Peak tracemalloc (MB)",
                        alpha=0.9, edgecolor="black", linewidth=0.3)
            else:
                return

        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=25, ha="right")
        ax2.set_ylabel("MB")
        ax2.set_title("Memory Breakdown (stacked)")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(loc="best", fontsize=9)

        fig.suptitle(title)
        cls.save_figure(fig, filename)



    #Accuracy vs total compute time (varies by dim)
    @classmethod
    def accuracy_vs_total_cost_curve(cls, df: pd.DataFrame, title: str, filename: str):
        plot_df = df.dropna(subset=["dim", "accuracy"]).copy()
        if plot_df.empty:
            return

        plot_df["total_time_s"] = plot_df["clf_train_time_s"].fillna(0) + plot_df["embed_time_s"].fillna(0)
        plot_df = plot_df.dropna(subset=["total_time_s"])
        if plot_df.empty:
            return

        fig, ax = plt.subplots(figsize=Config.FIG_SIZE_WIDE)
        for method, g in plot_df.groupby("method"):
            
            g = g.sort_values("dim")
            ax.plot(g["total_time_s"], g["accuracy"], marker="o", linewidth=2, label=method, alpha=0.9)
            
            for _, row in g.iterrows():
                if pd.notna(row["dim"]):
                    ax.annotate(
                        f"{int(row['dim'])}",
                        (row["total_time_s"], row["accuracy"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, alpha=0.8)
                    
        ax.set_xlabel("Total Compute Time (s)  [embed + classifier train]")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        cls.save_figure(fig, filename)



def select_rows_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if Config.USE_FIXED_DIM:
        out = df[df["dim"] == Config.FIXED_DIM].copy()
    elif Config.USE_MAX_DIM_PER_METHOD:
        idx = df.groupby("method")["dim"].transform("max") == df["dim"]
        out = df[idx].copy()
    else:
        mx = df["dim"].max()
        out = df[df["dim"] == mx].copy()

    if out.empty:
        return out

    return out.groupby("method", as_index=False).mean(numeric_only=True)



def generate_all_plots(agg: pd.DataFrame):
    datasets = sorted(agg["dataset"].dropna().unique().tolist())
    avg_by_method_dim = agg.groupby(["method", "dim"], as_index=False).mean(numeric_only=True)

    if Config.USE_FIXED_DIM:
        dim_note = f"(dim={Config.FIXED_DIM})"
    elif Config.USE_MAX_DIM_PER_METHOD:
        dim_note = "(each method at max dim)"
    else:
        dim_note = ""

    # OVERALL
    if Config.MAKE_AVG_PLOTS:
        overall_chosen = select_rows_for_comparison(avg_by_method_dim.assign(dataset="AVG"))
        if not overall_chosen.empty:
            
            Plotter.grouped_metric_bars(overall_chosen, title=f"Overall Performance Comparison {dim_note}", filename="BAR_OVERALL_ACC_F1_AUC.png")
            Plotter.stacked_time_memory_bars(overall_chosen,title=f"Overall Resource Usage (stacked) {dim_note}", filename="STACKED_OVERALL_time_memory.png")

        for metric in ["accuracy", "f1", "auc"]:
            Plotter.metric_vs_dimension_lines( avg_by_method_dim.assign(dataset="AVG"),metric_col=metric, title=f"Average {metric.title()} vs Embedding Dimension",  filename=f"LINE_AVG_{metric}_vs_dim.png",)

        for col, label in [("clf_train_time_s", "Classifier Training Time"), ("embed_time_s", "Embedding Generation Time"), ("peak_tracemalloc_mb", "Peak Memory Usage"), ]:
            if col in avg_by_method_dim.columns:
               
                Plotter.metric_vs_dimension_lines(
                    avg_by_method_dim.assign(dataset="AVG"),
                    metric_col=col,
                    title=f"Average {label} vs Embedding Dimension",
                    filename=f"LINE_AVG_{col}_vs_dim.png", )

        Plotter.accuracy_efficiency_scatter(
            avg_by_method_dim.assign(dataset="AVG"),
            time_col="clf_train_time_s",
            title="Accuracy vs Training Time Trade-off (Average)",
            filename="SCATTER_AVG_acc_vs_train_time.png", )

        Plotter.accuracy_vs_compute_cost_dual(
            avg_by_method_dim.assign(dataset="AVG"),
            title="Accuracy vs Compute Cost by Embedding Dimension (Average)",
            filename="DUAL_AVG_acc_vs_compute_cost.png",)

        Plotter.accuracy_vs_compute_cost_pareto(
            avg_by_method_dim.assign(dataset="AVG"),
            title="Pareto Frontier: Accuracy vs Compute Cost (Average)",
            filename="PARETO_AVG_acc_vs_compute_cost.png",)

        Plotter.accuracy_vs_compute_cost_faceted(
            avg_by_method_dim.assign(dataset="AVG"),
            title_prefix="Accuracy vs Individual Cost Metrics (Average)",
            filename="FACETED_AVG_acc_vs_costs.png",)

        Plotter.accuracy_vs_total_cost_curve(
            avg_by_method_dim.assign(dataset="AVG"),
            title="Accuracy vs Total Compute Time (Average, varies by dimension)",
            filename="COSTCURVE_AVG_acc_vs_total_time.png",)

        Plotter.performance_heatmap(
            avg_by_method_dim.assign(dataset="AVG"),
            metric_col="accuracy",
            title="Average Accuracy Heatmap (Method × Dimension)",
            filename="HEATMAP_AVG_accuracy.png",)

    if Config.MAKE_PER_DATASET:
        print("\nCreating per-dataset plots...")
        for ds in datasets:
            print(f"  Processing {ds}...")
            df_ds = agg[agg["dataset"] == ds].copy()
            chosen = select_rows_for_comparison(df_ds)
            if chosen.empty:
                continue

            Plotter.grouped_metric_bars(
                chosen,
                title=f"{ds}: Performance Comparison {dim_note}",
                filename=f"BAR_{ds}_ACC_F1_AUC.png", )

            Plotter.stacked_time_memory_bars(
                chosen,
                title=f"{ds}: Resource Usage (stacked) {dim_note}",
                filename=f"STACKED_{ds}_time_memory.png", )

            for metric in ["accuracy", "f1", "auc"]:
                Plotter.metric_vs_dimension_lines(
                    df_ds,
                    metric_col=metric,
                    title=f"{ds}: {metric.title()} vs Embedding Dimension",
                    filename=f"LINE_{ds}_{metric}_vs_dim.png",)

            Plotter.accuracy_efficiency_scatter(
                df_ds,
                time_col="clf_train_time_s",
                title=f"{ds}: Accuracy vs Training Time",
                filename=f"SCATTER_{ds}_acc_vs_train_time.png", )

            Plotter.memory_usage_bars(
                chosen,
                title=f"{ds}: Peak Memory Usage {dim_note}",
                filename=f"BAR_{ds}_memory.png",)

            Plotter.accuracy_vs_compute_cost_dual(
                df_ds,
                title=f"{ds}: Accuracy vs Compute Cost by Dimension",
                filename=f"DUAL_{ds}_acc_vs_compute_cost.png",)

            Plotter.accuracy_vs_compute_cost_pareto(
                df_ds,
                title=f"{ds}: Pareto Frontier - Accuracy vs Compute Cost",
                filename=f"PARETO_{ds}_acc_vs_compute_cost.png",)

            Plotter.accuracy_vs_compute_cost_faceted(
                df_ds,
                title_prefix=f"{ds}: Accuracy vs Individual Cost Metrics",
                filename=f"FACETED_{ds}_acc_vs_costs.png",)
            
            Plotter.accuracy_vs_total_cost_curve(
                df_ds,
                title=f"{ds}: Accuracy vs Total Compute Time (varies by dimension)",
                filename=f"COSTCURVE_{ds}_acc_vs_total_time.png",)
            


def main():
    print("=" * 60)
    print("Graph Embedding Classification Analysis")
    print("=" * 60)

    Config.setup()

    print("\nLoading data files...")
    gin = DataProcessor.read_csv_safe(Config.GIN_PATH)
    g2v_cls = DataProcessor.read_csv_safe(Config.G2V_CLS_PATH)
    netlsd_cls = DataProcessor.read_csv_safe(Config.NETLSD_CLS_PATH)
    g2v_metrics = DataProcessor.read_csv_safe(Config.G2V_METRICS_PATH)
    netlsd_metrics = DataProcessor.read_csv_safe(Config.NETLSD_METRICS_PATH)
    metrics_summary = DataProcessor.read_csv_safe(Config.METRICS_SUMMARY_PATH)

    print("Processing data...")
    all_rows = []
    if gin is not None:
        print("  - GIN data found")
        all_rows.append(DataProcessor.process_gin_data(gin))
    elif metrics_summary is not None:
        print("  - Using metrics_summary as GIN fallback")
        all_rows.append(DataProcessor.process_gin_data(metrics_summary))

    if g2v_cls is not None:
        print("  - Graph2Vec data found")
        all_rows.append(DataProcessor.process_unsupervised_data(g2v_cls, g2v_metrics, "Graph2Vec"))

    if netlsd_cls is not None:
        print("  - NetLSD data found")
        all_rows.append(DataProcessor.process_unsupervised_data(netlsd_cls, netlsd_metrics, "NetLSD"))

    if not all_rows:
        raise SystemExit("ERROR: No input data files found!")

    all_data = pd.concat(all_rows, ignore_index=True)
    all_data["method"] = all_data["method"].apply(DataProcessor.standardize_method_name)

    print("\nAggregating results...")
    agg = all_data.groupby(["dataset", "method", "dim"], as_index=False).mean(numeric_only=True)

    unified_path = Config.OUT_DIR / "unified_results_mean.csv"
    agg.to_csv(unified_path, index=False)
    print(f"Saved unified results to: {unified_path}")

    generate_all_plots(agg)
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Output directory: {Config.OUT_DIR.resolve()}")
    print(f"Total plots generated: {len(list(Config.OUT_DIR.glob('*.png')))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
