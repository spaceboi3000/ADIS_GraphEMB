from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
import umap  # pip install umap-learn



DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]

GIN_OUT_ROOT = Path("../embeddings/embeddings_gin")
METHOD_NAME = "GIN"

SPLITS = ["train", "val", "test"]  


VIS_BASE = Path("./cluster_visualizations")
VIS_BASE.mkdir(exist_ok=True)

RANDOM_STATE = 42
KMEANS_N_INIT = 80
KMEANS_MAX_ITER = 1000



def load_all_splits(run_dir: Path, splits) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concat train/val/test from:
        <run_dir>/<split>_embeddings_wide.csv
    """
    dfs: List[pd.DataFrame] = []
    for split in splits:
        csv_path = run_dir / f"{split}_embeddings_wide.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "label" not in df.columns:
                raise ValueError(f"'label' column not found in {csv_path}")
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No split CSVs found in {run_dir} for {splits}")
    df_all = pd.concat(dfs, ignore_index=True)
    y = df_all["label"].to_numpy().astype(int)
    X = df_all.drop(columns=["label"]).to_numpy().astype(np.float32)
    return X, y


def run_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    km = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    return km.fit_predict(X)


def run_spectral(X: np.ndarray, n_clusters: int) -> np.ndarray:
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    )
    return sc.fit_predict(X)


def compute_tsne(X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    perplexity = min(30, max(5, n_samples // 10))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def compute_umap(X: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(X)


def scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    legend_title: str,
):
    plt.figure(figsize=(6, 5))
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.8,
            label=str(lab),
        )

    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(title=legend_title, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    for dataset in DATASETS:
        ds_root = GIN_OUT_ROOT / dataset
        if not ds_root.exists():
            print(f"[WARN] Dataset dir not found: {ds_root}")
            continue

        run_dirs = sorted(
            [p for p in ds_root.iterdir() if p.is_dir() and p.name.startswith("gin_dim")]
        )
        if not run_dirs:
            print(f"[WARN] No GIN run dirs under {ds_root}")
            continue

        print(f"\n=== {METHOD_NAME} t-SNE / UMAP :: {dataset} ===")

        for run_dir in run_dirs:
            try:
                X, y = load_all_splits(run_dir, SPLITS)
            except FileNotFoundError as e:
                print(f"  [WARN] {run_dir.name}: {e}")
                continue

            if X.size == 0:
                print(f"  [WARN] {run_dir.name}: empty embeddings")
                continue

            n_samples, emb_dim = X.shape
            n_clusters = len(np.unique(y))
            print(
                f"  Run={run_dir.name}, splits={SPLITS} -> X.shape={X.shape}, "
                f"n_clusters={n_clusters}, labels={sorted(np.unique(y).tolist())}"
            )

            X_vis = X
            y_vis = y

            # Cluster assignments
            print("    Running KMeans...")
            y_kmeans = run_kmeans(X_vis, n_clusters)

            print("    Running Spectral clustering...")
            try:
                y_spectral = run_spectral(X_vis, n_clusters)
            except Exception as e:
                print(f"    [WARN] Spectral clustering failed for {run_dir.name}: {e}")
                y_spectral = None

            # 2D embeddings
            print("    Computing t-SNE...")
            X_tsne = compute_tsne(X_vis)

            print("    Computing UMAP...")
            X_umap = compute_umap(X_vis)

            # Visualization directory:
            # cluster_visualizations/GIN/<dataset>/<run_name>/
            out_dir = VIS_BASE / METHOD_NAME / dataset / run_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            tsne_label_path = out_dir / f"{METHOD_NAME}_tsne_by_label.png"
            tsne_km_path    = out_dir / f"{METHOD_NAME}_tsne_by_kmeans.png"
            tsne_sp_path    = out_dir / f"{METHOD_NAME}_tsne_by_spectral.png"

            umap_label_path = out_dir / f"{METHOD_NAME}_umap_by_label.png"
            umap_km_path    = out_dir / f"{METHOD_NAME}_umap_by_kmeans.png"
            umap_sp_path    = out_dir / f"{METHOD_NAME}_umap_by_spectral.png"

            # True labels
            scatter_plot(
                X_tsne,
                y_vis,
                f"{METHOD_NAME} – {dataset} – {run_dir.name} – t-SNE (labels)",
                tsne_label_path,
                "Label",
            )
            scatter_plot(
                X_umap,
                y_vis,
                f"{METHOD_NAME} – {dataset} – {run_dir.name} – UMAP (labels)",
                umap_label_path,
                "Label",
            )

            # KMeans
            scatter_plot(
                X_tsne,
                y_kmeans,
                f"{METHOD_NAME} – {dataset} – {run_dir.name} – t-SNE (KMeans)",
                tsne_km_path,
                "Cluster (KMeans)",
            )
            scatter_plot(
                X_umap,
                y_kmeans,
                f"{METHOD_NAME} – {dataset} – {run_dir.name} – UMAP (KMeans)",
                umap_km_path,
                "Cluster (KMeans)",
            )

            # Spectral
            if y_spectral is not None:
                scatter_plot(
                    X_tsne,
                    y_spectral,
                    f"{METHOD_NAME} – {dataset} – {run_dir.name} – t-SNE (Spectral)",
                    tsne_sp_path,
                    "Cluster (Spectral)",
                )
                scatter_plot(
                    X_umap,
                    y_spectral,
                    f"{METHOD_NAME} – {dataset} – {run_dir.name} – UMAP (Spectral)",
                    umap_sp_path,
                    "Cluster (Spectral)",
                )

            print("    Saved:")
            print(f"      {tsne_label_path}")
            print(f"      {tsne_km_path}")
            if y_spectral is not None:
                print(f"      {tsne_sp_path}")
            print(f"      {umap_label_path}")
            print(f"      {umap_km_path}")
            if y_spectral is not None:
                print(f"      {umap_sp_path}")


if __name__ == "__main__":
    main()
