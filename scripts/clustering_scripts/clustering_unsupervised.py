from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment


# ================== CONFIG ==================

DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]

# Base directory where all embeddings live
BASE_EMB_DIR = Path("../embeddings")

METHODS = {
    "Graph2Vec": {
        "out_dir": BASE_EMB_DIR / "embeddings_graph2vec",
        "has_dims": True,
        "dims": [64, 128, 256],
    },
    "NetLSD": {
        "out_dir": BASE_EMB_DIR / "embeddings_netlsd",
        "has_dims": True,
        "dims": [64, 128, 256],
    },
}

RANDOM_STATE = 42
KMEANS_N_INIT = 10  # was 10
KMEANS_MAX_ITER = 3000
# ===========================================


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Clustering accuracy via Hungarian matching.
    """
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / cm.sum()
    return float(acc)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict:
    """
    Common metric computation for any clustering algorithm.
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)
    # silhouette can fail if only 1 cluster or weird label config
    if len(np.unique(y_pred)) > 1 and X.shape[0] > len(np.unique(y_pred)):
        sil = silhouette_score(X, y_pred)
    else:
        sil = float("nan")
    return dict(nmi=nmi, ari=ari, acc=acc, silhouette=sil)


def run_kmeans_for_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
) -> Dict:
    """
    Run KMeans and compute metrics for given embeddings X and labels y.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    y_pred = kmeans.fit_predict(X)
    return _compute_metrics(y, y_pred, X)


def run_spectral_for_embeddings(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
) -> Dict:
    """
    Run Spectral Clustering and compute metrics.
    """
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    )
    y_pred = sc.fit_predict(X)
    return _compute_metrics(y, y_pred, X)


def load_embeddings_csv(
    method: str,
    method_cfg: Dict,
    dataset: str,
    dim: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and labels for a given method/dataset/(dim).
    """
    out_dir = method_cfg["out_dir"]
    has_dims = method_cfg["has_dims"]

    if has_dims:
        if dim is None:
            raise ValueError(f"Method {method} expects a dim but got None.")
        emb_dir = out_dir / dataset / f"dim{dim}"
    else:
        emb_dir = out_dir / dataset

    csv_path = emb_dir / f"{method}_embeddings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    y = df["label"].to_numpy().astype(int)
    X = df.drop(columns=["label"]).to_numpy().astype(np.float32)
    return X, y


def main():
    for method, cfg in METHODS.items():
        print(f"\n================== {method} CLUSTERING ==================")
        rows: List[Dict] = []

        for dataset in DATASETS:
            for dim in cfg["dims"]:
                try:
                    X, y = load_embeddings_csv(method, cfg, dataset, dim)
                except FileNotFoundError as e:
                    print(f"[WARN] {method} {dataset} dim={dim}: {e}")
                    continue

                n_samples, emb_dim = X.shape
                n_clusters = len(np.unique(y))

                print(
                    f"\n  {dataset} dim={dim} -> X.shape={X.shape}, "
                    f"n_clusters={n_clusters}, labels={sorted(np.unique(y).tolist())}"
                )

                # --------- KMEANS ----------
                km_metrics = run_kmeans_for_embeddings(X, y, n_clusters)
                print(
                    f"    [KMEANS]  NMI={km_metrics['nmi']:.4f}, "
                    f"ARI={km_metrics['ari']:.4f}, "
                    f"ACC={km_metrics['acc']:.4f}, "
                    f"SIL={km_metrics['silhouette']:.4f}"
                )
                rows.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        dim=dim if dim is not None else emb_dim,
                        n_graphs=n_samples,
                        n_clusters=n_clusters,
                        cluster_alg="kmeans",
                        nmi=round(km_metrics["nmi"], 6),
                        ari=round(km_metrics["ari"], 6),
                        acc=round(km_metrics["acc"], 6),
                        silhouette=round(km_metrics["silhouette"], 6),
                    )
                )

                # --------- SPECTRAL ----------
                try:
                    sp_metrics = run_spectral_for_embeddings(X, y, n_clusters)
                    print(
                        f"    [SPECTRAL] NMI={sp_metrics['nmi']:.4f}, "
                        f"ARI={sp_metrics['ari']:.4f}, "
                        f"ACC={sp_metrics['acc']:.4f}, "
                        f"SIL={sp_metrics['silhouette']:.4f}"
                    )
                    rows.append(
                        dict(
                            dataset=dataset,
                            method=method,
                            dim=dim if dim is not None else emb_dim,
                            n_graphs=n_samples,
                            n_clusters=n_clusters,
                            cluster_alg="spectral",
                            nmi=round(sp_metrics["nmi"], 6),
                            ari=round(sp_metrics["ari"], 6),
                            acc=round(sp_metrics["acc"], 6),
                            silhouette=round(sp_metrics["silhouette"], 6),
                        )
                    )
                except Exception as e:
                    print(f"    [SPECTRAL] failed: {e}")

        if not rows:
            print(f"\nNo results to save for {method}.")
            continue

        # One CSV per method (overwrite each run)
        out_dir = cfg["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / f"cluster_results_{method.lower()}.csv"

        df_new = pd.DataFrame(rows)
        df_new.to_csv(results_path, index=False)
        print(f"\nClustering metrics for {method} written to {results_path}")


if __name__ == "__main__":
    main()
