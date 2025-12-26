from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

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

GIN_OUT_ROOT = Path("../permutated_embeddings/embeddings_gin")

METHOD_NAME = "GIN"
SPLITS = ["train", "val", "test"] 

RANDOM_STATE = 42
KMEANS_N_INIT = 80
KMEANS_MAX_ITER = 1000
# ===========================================


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Clustering accuracy via Hungarian matching.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 0:
        return float("nan")
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / cm.sum()
    return float(acc)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Dict:
    """
    Shared metric computation (NMI, ARI, ACC, silhouette).
    """
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    acc = clustering_accuracy(y_true, y_pred)
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
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    )
    y_pred = sc.fit_predict(X)
    return _compute_metrics(y, y_pred, X)


def load_all_splits(run_dir: Path, splits) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concat train/val/test from:
        <run_dir>/<split>_embeddings_wide.csv
    Format:
        label, dim0, dim1, ...
    """
    dfs = []
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


def main():
    all_rows: List[Dict] = []

    for dataset in DATASETS:
        ds_root = GIN_OUT_ROOT / dataset
        if not ds_root.exists():
            print(f"[WARN] Dataset dir not found: {ds_root}")
            continue

        # Run dirs: gin_dim*_ep*_lr*_drop*_...
        run_dirs = sorted(
            [p for p in ds_root.iterdir() if p.is_dir() and p.name.startswith("gin_dim")]
        )

        if not run_dirs:
            print(f"[WARN] No GIN run dirs under {ds_root}")
            continue

        print(f"\n=== {METHOD_NAME} CLUSTERING :: {dataset} ===")
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
                f"\n  Run={run_dir.name}, splits={SPLITS} -> X.shape={X.shape}, "
                f"n_clusters={n_clusters}, labels={sorted(np.unique(y).tolist())}"
            )

            # --- KMEANS ---
            km_metrics = run_kmeans_for_embeddings(X, y, n_clusters)
            print(
                f"    [KMEANS]   NMI={km_metrics['nmi']:.4f}, "
                f"ARI={km_metrics['ari']:.4f}, "
                f"ACC={km_metrics['acc']:.4f}, "
                f"SIL={km_metrics['silhouette']:.4f}"
            )
            all_rows.append(
                dict(
                    dataset=dataset,
                    method=METHOD_NAME,
                    run_name=run_dir.name,
                    split="all",
                    dim=emb_dim,
                    n_graphs=n_samples,
                    n_clusters=n_clusters,
                    cluster_alg="kmeans",
                    nmi=round(km_metrics["nmi"], 6),
                    ari=round(km_metrics["ari"], 6),
                    acc=round(km_metrics["acc"], 6),
                    silhouette=round(km_metrics["silhouette"], 6),
                )
            )

            # --- SPECTRAL ---
            try:
                sp_metrics = run_spectral_for_embeddings(X, y, n_clusters)
                print(
                    f"    [SPECTRAL] NMI={sp_metrics['nmi']:.4f}, "
                    f"ARI={sp_metrics['ari']:.4f}, "
                    f"ACC={sp_metrics['acc']:.4f}, "
                    f"SIL={sp_metrics['silhouette']:.4f}"
                )
                all_rows.append(
                    dict(
                        dataset=dataset,
                        method=METHOD_NAME,
                        run_name=run_dir.name,
                        split="all",
                        dim=emb_dim,
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
                print(f"    [SPECTRAL] failed for {run_dir.name}: {e}")

    if not all_rows:
        print("\nNo GIN clustering results to save.")
        return

    results_path = GIN_OUT_ROOT / "cluster_results_gin.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(results_path, index=False)
    print(f"\n[OK] GIN clustering metrics written to {results_path}")


if __name__ == "__main__":
    main()
