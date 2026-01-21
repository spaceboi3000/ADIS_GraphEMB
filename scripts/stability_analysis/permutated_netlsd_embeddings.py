from __future__ import annotations
import time, os, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.decomposition import PCA

import torch
from torch_geometric.utils import to_networkx

#permutated datas
PERMUTATED_ROOT = Path("../permutated_DATASETS")

DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
TARGET_DIMS = [64, 128, 256]

OUT_DIR = Path("../permutated_embeddings/permutated_netlsd")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = OUT_DIR / "permutated_metrics.csv"

# Length of NetLSD heat kernel signature before dimensionality reduction
NETLSD_SCALE_N = 250 


def load_permutated_dataset(name: str, perm_root: Path) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Load a permutated dataset list saved via torch.save(permutated_list, ...).
    Each element is a PyG Data with perturbed edges and maybe shuffled node features.
    Convert each to a NetworkX graph and attach node features under 'feat' if present.
    """
    perm_path = perm_root / name / f"{name}_permutated.pt"
    if not perm_path.exists():
        raise FileNotFoundError(f"Permutated dataset file not found: {perm_path}")

    # FutureWarning is fine; you're loading your own files
    data_list = torch.load(perm_path, weights_only=False) 
    graphs: List[nx.Graph] = []
    labels = []

    for data in data_list:
        g = to_networkx(data, to_undirected=True)

        # Add node features as "feat" if they exist
        if getattr(data, "x", None) is not None:
            x_np = data.x.cpu().numpy()
            for i, (_, d) in enumerate(g.nodes(data=True)):
                d["feat"] = x_np[i].tolist()

        graphs.append(g)
        labels.append(int(data.y.item()))

    return graphs, np.array(labels)


def ensure_dimensionality(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Make sure NetLSD signatures X (n x d) become (n x target_dim):
    - If d > target_dim: PCA
    - If d < target_dim: zero-pad
    """
    n, d = X.shape
    if d == target_dim:
        return X.astype(np.float32)
    if d > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        return pca.fit_transform(X).astype(np.float32)
    # d < target_dim -> pad
    out = np.zeros((n, target_dim), dtype=np.float32)
    out[:, :d] = X.astype(np.float32)
    return out


def _dense_netlsd_signatures(graphs: List[nx.Graph], scale_n: int = 250) -> np.ndarray:
    """
    Dense NetLSD-style signatures using normalized Laplacian eigenvalues directly.
    We implement the normalized Laplacian ourselves to avoid SciPy completely.

    L = I - D^{-1/2} A D^{-1/2}
    """
    scales = np.logspace(-2, 2, scale_n, base=10.0, dtype=np.float64)
    embs = []

    for g in graphs:
        # Adjacency matrix (dense)
        # Order of nodes is deterministic in nx.to_numpy_array
        A = nx.to_numpy_array(g, dtype=np.float64)
        n = A.shape[0]

        if n == 0:
            embs.append(np.zeros(scale_n, dtype=np.float32))
            continue

        # Degree vector
        deg = A.sum(axis=1)
        # D^{-1/2}, careful with zeros
        with np.errstate(divide="ignore"):
            inv_sqrt_deg = 1.0 / np.sqrt(deg)
        inv_sqrt_deg[np.isinf(inv_sqrt_deg)] = 0.0
        inv_sqrt_deg[np.isnan(inv_sqrt_deg)] = 0.0

        # Construct normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_half = np.diag(inv_sqrt_deg)
        L = np.eye(n, dtype=np.float64) - D_half @ A @ D_half

        try:
            evals = np.linalg.eigvalsh(L)
        except np.linalg.LinAlgError:
            
            L = L + 1e-10 * np.eye(n, dtype=np.float64) #small regularization in case of numerical issues
            evals = np.linalg.eigvalsh(L)

        # Heat kernel signature
        sig = np.exp(-np.outer(scales, evals)).sum(axis=1)
        embs.append(sig.astype(np.float32))

    return np.vstack(embs)


def save_embeddings(dataset: str, method: str, dim: int, X: np.ndarray, y: np.ndarray):

    ds_dir = OUT_DIR / dataset / f"dim{dim}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / f"{method}_embeddings.npy", X.astype(np.float32))
    
    cols = [f"dim{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "label", y.astype(int))
    df.to_csv(ds_dir / f"{method}_embeddings.csv", index=False)



def run_dense_netlsd_for_permutated_dataset(name: str):
    print(f"\nPermutated NetLSD (dense, no SciPy) :: {name}")

    graphs, y = load_permutated_dataset(name, PERMUTATED_ROOT)
    print(f"Loaded {len(graphs)} permutated graphs; classes: {sorted(set(y.tolist()))}")

    import tracemalloc, psutil
    proc = psutil.Process(os.getpid())

    metrics_rows = []

    #Compute raw NetLSD signatures once
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_fit0 = time.time()
        X_raw = _dense_netlsd_signatures(graphs, scale_n=NETLSD_SCALE_N)
        t_fit1 = time.time()

    X_raw = np.nan_to_num(X_raw, copy=False)
    fit_time = t_fit1 - t_fit0

    for dim in TARGET_DIMS:
        tracemalloc.start()
        rss_before = proc.memory_info().rss
        t0 = time.time()

        X = ensure_dimensionality(X_raw, dim)

        t1 = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss

        pca_time = t1 - t0
        total_time = fit_time + pca_time
        peak_mb = peak / (1024 ** 2)
        rss_before_mb = rss_before / (1024 ** 2)
        rss_after_mb = rss_after / (1024 ** 2)

        print(
            f"dim={dim} -> raw {X_raw.shape} -> {X.shape}, "
            f"fit {fit_time:.2f}s, PCA/pad {pca_time:.2f}s, peak_mem {peak_mb:.1f} MB"
        )

        save_embeddings(name, "NetLSD", dim, X, y)

        metrics_rows.append({
            "dataset": name,
            "method": "NetLSD_permutated_dense",
            "dim": dim,
            "n_graphs": len(graphs),
            "fit_time_s": round(fit_time, 4),
            "pca_time_s": round(pca_time, 4),
            "total_time_s": round(total_time, 4),
            "rss_before_mb": round(rss_before_mb, 2),
            "rss_after_mb": round(rss_after_mb, 2),
            "peak_tracemalloc_mb": round(peak_mb, 2),
            "raw_sig_len": X_raw.shape[1],
        })

    return metrics_rows


if __name__ == "__main__":
    print("PERMUTATED_ROOT:", PERMUTATED_ROOT)
    print("OUT_DIR:", OUT_DIR)

    all_metrics = []

    for ds in DATASETS:
        try:
            rows = run_dense_netlsd_for_permutated_dataset(ds)
            all_metrics.extend(rows)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")

    if all_metrics:
        mdf = pd.DataFrame(all_metrics)
        if METRICS_PATH.exists():
            old = pd.read_csv(METRICS_PATH)
            mdf = pd.concat([old, mdf], ignore_index=True)
        mdf.to_csv(METRICS_PATH, index=False)
        print(f"\nMetrics written to {METRICS_PATH}")
    else:
        print("\nNo metrics collected (all runs failed?).")
