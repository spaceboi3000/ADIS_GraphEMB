from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
import umap  # pip install umap-learn


# ---- Config ----
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
EMBEDDING_DIMS = [64, 128, 256]

BASE_EMB_DIR = Path("../embeddings")  # Graph2Vec + NetLSD live here

METHODS: Dict[str, Dict] = {
    "Graph2Vec": {
        "emb_dir": BASE_EMB_DIR / "embeddings_graph2vec",
        "dims": EMBEDDING_DIMS,
    },
    "NetLSD": {
        "emb_dir": BASE_EMB_DIR / "embeddings_netlsd",
        "dims": EMBEDDING_DIMS,
    },
}

# Visualization folder OUTSIDE embeddings/
VIS_BASE = Path("./cluster_visualizations")
VIS_BASE.mkdir(exist_ok=True)

RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 3000


def load_embeddings_csv(
    method: str,
    method_cfg: Dict,
    dataset: str,
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from:
        ../embeddings/<method_emb_dir>/<dataset>/dim<dim>/<Method>_embeddings.csv
    """
    emb_dir = method_cfg["emb_dir"] / dataset / f"dim{dim}"
    csv_path = emb_dir / f"{method}_embeddings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    y = df["label"].to_numpy().astype(int)
    X = df.drop(columns=["label"]).to_numpy().astype(np.float32)
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
    for method, cfg in METHODS.items():
        print(f"\n============== {method} VISUALIZATIONS ==============")

        for dataset in DATASETS:
            print(f"\n=== {method} :: {dataset} ===")

            for dim in cfg["dims"]:
                try:
                    X, y = load_embeddings_csv(method, cfg, dataset, dim)
                except FileNotFoundError as e:
                    print(f"[WARN] {method} {dataset} dim={dim}: {e}")
                    continue

                n_samples, _ = X.shape
                n_clusters = len(np.unique(y))

                X_vis = X
                y_vis = y

                # 1) Cluster assignments
                print("   Running KMeans...")
                y_kmeans = run_kmeans(X_vis, n_clusters)

                print("   Running Spectral clustering...")
                try:
                    y_spectral = run_spectral(X_vis, n_clusters)
                except Exception as e:
                    print(f"   [WARN] Spectral clustering failed for {method} {dataset} dim={dim}: {e}")
                    y_spectral = None

                # 2) t-SNE / UMAP (only once)
                print("   Computing t-SNE...")
                X_tsne = compute_tsne(X_vis)

                print("   Computing UMAP...")
                X_umap = compute_umap(X_vis)

                # Create visualization directory:
                # cluster_visualizations/Graph2Vec/MUTAG/dim64/
                ds_dir = VIS_BASE / method / dataset / f"dim{dim}"
                ds_dir.mkdir(parents=True, exist_ok=True)

                # Filenames: labels, kmeans, spectral
                tsne_label_path = ds_dir / f"{method}_dim{dim}_tsne_by_label.png"
                tsne_km_path    = ds_dir / f"{method}_dim{dim}_tsne_by_kmeans.png"
                tsne_sp_path    = ds_dir / f"{method}_dim{dim}_tsne_by_spectral.png"

                umap_label_path = ds_dir / f"{method}_dim{dim}_umap_by_label.png"
                umap_km_path    = ds_dir / f"{method}_dim{dim}_umap_by_kmeans.png"
                umap_sp_path    = ds_dir / f"{method}_dim{dim}_umap_by_spectral.png"

                # 3) Save plots

                # True labels
                scatter_plot(
                    X_tsne,
                    y_vis,
                    f"{method} - t-SNE (true labels)",
                    tsne_label_path,
                    "Label",
                )
                scatter_plot(
                    X_umap,
                    y_vis,
                    f"{method} - UMAP (true labels)",
                    umap_label_path,
                    "Label",
                )

                # KMeans clusters
                scatter_plot(
                    X_tsne,
                    y_kmeans,
                    f"{method} - t-SNE (KMeans clusters)",
                    tsne_km_path,
                    "Cluster (KMeans)",
                )
                scatter_plot(
                    X_umap,
                    y_kmeans,
                    f"{method} - UMAP (KMeans clusters)",
                    umap_km_path,
                    "Cluster (KMeans)",
                )

                # Spectral clusters (only if it worked)
                if y_spectral is not None:
                    scatter_plot(
                        X_tsne,
                        y_spectral,
                        f"{method} - t-SNE (Spectral clusters)",
                        tsne_sp_path,
                        "Cluster (Spectral)",
                    )
                    scatter_plot(
                        X_umap,
                        y_spectral,
                        f"{method} - UMAP (Spectral clusters)",
                        umap_sp_path,
                        "Cluster (Spectral)",
                    )

                print("   Saved:")
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
