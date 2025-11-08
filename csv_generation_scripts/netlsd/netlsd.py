from __future__ import annotations
import time, os, warnings
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

import networkx as nx
from karateclub import NetLSD
from sklearn.decomposition import PCA

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# Config 
DATASETS_ROOT = Path("../DATASETS")
STRICT_LOCAL_ONLY = False           
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
TARGET_DIMS = [64, 128, 256]  #dif dim
OUT_DIR = Path("./embeddings_netlsd")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NETLSD_PARAMS: Dict = dict(
    scale_min=None,
    scale_max=None,
    scale_n=None
)

# Helpers
def _tu_processed_paths(root: Path, name: str):
    return [root / name / "processed", root / "TUDataset" / name / "processed"]

def _assert_local_available(root: Path, name: str):
    if not STRICT_LOCAL_ONLY:
        return
    if not any(p.exists() for p in _tu_processed_paths(root, name)):
        raise FileNotFoundError(
            f"Local TU dataset not found for '{name}'. Checked: "
            + ", ".join(map(str, _tu_processed_paths(root, name)))
        )

def load_tu_dataset(name: str, root: Path) -> Tuple[List[nx.Graph], np.ndarray]:
    _assert_local_available(root, name)
    ds = TUDataset(root=str(root), name=name)
    graphs: List[nx.Graph] = []
    labels = []
    for data in ds:
        g = to_networkx(data, to_undirected=True)
        if getattr(data, "x", None) is not None:
            x_np = data.x.cpu().numpy()
            for i, (_, d) in enumerate(g.nodes(data=True)):
                d["feat"] = x_np[i].tolist()
        graphs.append(g)
        labels.append(int(data.y.item()))
    return graphs, np.array(labels)

def ensure_dimensionality(X: np.ndarray, target_dim: int) -> np.ndarray:
    n, d = X.shape
    if d == target_dim:
        return X.astype(np.float32)
    if d > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        return pca.fit_transform(X).astype(np.float32)
    out = np.zeros((n, target_dim), dtype=np.float32)
    out[:, :d] = X.astype(np.float32)
    return out

def _dense_netlsd_signatures(graphs: List[nx.Graph], scale_n: int = 250) -> np.ndarray:
    # Robust fallback: dense eigs on normalized Laplacian, log-spaced scales
    scales = np.logspace(-2, 2, scale_n, base=10.0, dtype=np.float64)
    embs = []
    for g in graphs:
        L = nx.normalized_laplacian_matrix(g).toarray().astype(np.float64, copy=False)
        if L.size == 0:
            embs.append(np.zeros(scale_n, dtype=np.float32))
            continue
        try:
            evals = np.linalg.eigvalsh(L)
        except np.linalg.LinAlgError:
            L = L + 1e-10 * np.eye(L.shape[0], dtype=np.float64)
            evals = np.linalg.eigvalsh(L)
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

# Main
def run_netlsd_for_dataset(name: str):
    print(f"\n=== NetLSD :: {name} ===")
    graphs, y = load_tu_dataset(name, DATASETS_ROOT)
    print(f"Loaded {len(graphs)} graphs; classes: {sorted(set(y.tolist()))}")

    import tracemalloc, psutil
    proc = psutil.Process(os.getpid())

    metrics_rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_raw = None
        t_fit0 = time.time()
        try:
            model = NetLSD(**{k: v for k, v in NETLSD_PARAMS.items() if v is not None})
            model.fit(graphs)
            X_raw = model.get_embedding()
        except Exception as e:
            print(f"[INFO] {name}: KarateClub NetLSD failed ({e}). Falling back to dense-eigs NetLSD.")
            scale_n = NETLSD_PARAMS.get("scale_n") or 250
            X_raw = _dense_netlsd_signatures(graphs, scale_n=scale_n)
        t_fit1 = time.time()
    X_raw = np.nan_to_num(X_raw, copy=False)

    for dim in TARGET_DIMS:
        tracemalloc.start()
        rss_before = proc.memory_info().rss
        t0 = time.time()
        X = ensure_dimensionality(X_raw, dim)
        t1 = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss

        fit_time = t_fit1 - t_fit0
        pca_time = t1 - t0
        total_time = fit_time + pca_time
        peak_mb = peak / (1024**2)
        rss_before_mb = rss_before / (1024**2)
        rss_after_mb = rss_after / (1024**2)

        print(f"dim={dim} -> raw {X_raw.shape} -> {X.shape}, fit {fit_time:.2f}s, PCA {pca_time:.2f}s, peak_mem {peak_mb:.1f} MB")
        save_embeddings(name, "NetLSD", dim, X, y)
        metrics_rows.append({
            "dataset": name,
            "method": "NetLSD",
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

    # Append metrics CSV
    mpath = OUT_DIR / "metrics_netlsd.csv"
    mdf = pd.DataFrame(metrics_rows)
    if mpath.exists():
        old = pd.read_csv(mpath)
        mdf = pd.concat([old, mdf], ignore_index=True)
    mdf.to_csv(mpath, index=False)
    print(f"Metrics appended to {mpath}")

if __name__ == "__main__":
    print("DATASETS_ROOT:", DATASETS_ROOT)
    print("STRICT_LOCAL_ONLY:", STRICT_LOCAL_ONLY)
    for ds in DATASETS:
        try:
            run_netlsd_for_dataset(ds)
        except Exception as e:
            print(f"[WARN] {ds}: {e}")
