"""
Task 1 (separate script): NetLSD with KarateClub on local TU datasets
--------------------------------------------------------------------
• Loads TU datasets (MUTAG, ENZYMES, IMDB-MULTI by default) from your local folder.
• Uses PyTorch Geometric to read graphs, converts to NetworkX.
• Fits NetLSD and saves embeddings (CSV + NPY) with labels.
• Applies PCA to enforce a common TARGET_DIM (since NetLSD's raw signature length can differ).

Setup (Colab or local):
  pip install karateclub==1.3.3 scikit-learn torch torchvision torchaudio
  # install PyG wheels matching your torch/CUDA from https://data.pyg.org/whl/

Folder expectation (STRICT_LOCAL_ONLY=True):
  DATASETS_ROOT/
    MUTAG/processed/
    ENZYMES/processed/
    IMDB-MULTI/processed/
  # Or under DATASETS_ROOT/TUDataset/<NAME>/processed/
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

import networkx as nx
from karateclub import NetLSD
from sklearn.decomposition import PCA
import warnings

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# ----------------- Config -----------------
DATASETS_ROOT = Path("../DATASETS")     # <-- change to your local TU datasets root
STRICT_LOCAL_ONLY = False               # don't download/process; fail if not present
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]  # change if your trio differs
TARGET_DIM = 128
OUT_DIR = Path("./embeddings_netlsd")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NetLSD config — leaving defaults None uses KarateClub's defaults
# NetLSD config — KarateClub's NetLSD does not accept a 'k' parameter in this version.
NETLSD_PARAMS: Dict = dict(
    scale_min=None,
    scale_max=None,
    scale_n=None,
)

# --------------- Helpers -----------------

def _dense_netlsd_signatures(graphs: List[nx.Graph], scale_n: int = 250) -> np.ndarray:
    """Fallback NetLSD: compute heat trace signatures per-graph using dense eigendecomposition.
    This is robust for TU datasets and avoids sparse-eigs 'k' issues.
    """
    # Use log-spaced time scales similar to common NetLSD defaults
    scales = np.logspace(-2, 2, scale_n, base=10.0, dtype=np.float64)
    embs = []
    for g in graphs:
        # Build normalized Laplacian (dense)
        L = nx.normalized_laplacian_matrix(g).toarray().astype(np.float64, copy=False)
        # Handle empty graphs safely
        if L.size == 0:
            embs.append(np.zeros(scale_n, dtype=np.float64))
            continue
        # Eigendecomposition
        try:
            evals = np.linalg.eigvalsh(L)
        except np.linalg.LinAlgError:
            # Add tiny jitter on diagonal and retry
            L = L + 1e-10 * np.eye(L.shape[0], dtype=np.float64)
            evals = np.linalg.eigvalsh(L)
        # Heat trace: sum(exp(-t * lambda_i)) for each scale t
        sig = np.exp(-np.outer(scales, evals)).sum(axis=1)
        embs.append(sig.astype(np.float32))
    return np.vstack(embs)

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
        return X
    if d > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        return pca.fit_transform(X)
    # pad if lower
    out = np.zeros((n, target_dim), dtype=X.dtype)
    out[:, :d] = X
    return out

def save_embeddings(dataset: str, method: str, X: np.ndarray, y: np.ndarray):
    ds_dir = OUT_DIR / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / f"{method}_embeddings.npy", X)
    df = pd.DataFrame(X)
    df.insert(0, "label", y)
    df.to_csv(ds_dir / f"{method}_embeddings.csv", index=False)

# --------------- Main -----------------

def run_netlsd_for_dataset(name: str):
    print(f"=== NetLSD :: {name} ===")
    graphs, y = load_tu_dataset(name, DATASETS_ROOT)
    print(f"Loaded {len(graphs)} graphs; classes: {sorted(set(y.tolist()))}")

    t0 = time.time()
    X_raw = None
    # Try KarateClub NetLSD first
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence scipy/eigs warnings
            model = NetLSD(**{k: v for k, v in NETLSD_PARAMS.items() if v is not None})
            model.fit(graphs)
            X_raw = model.get_embedding()
    except Exception as e:
        print(f"[INFO] {name}: KarateClub NetLSD failed ({e}). Falling back to dense-eigs NetLSD.")

    if X_raw is None:
        X_raw = _dense_netlsd_signatures(graphs, scale_n=NETLSD_PARAMS.get("scale_n", 250) or 250)

    # Clean and enforce target dim
    X_raw = np.nan_to_num(X_raw, copy=False)
    X = ensure_dimensionality(X_raw, TARGET_DIM).astype(np.float32)
    dt = time.time() - t0
    print(f"NetLSD produced raw shape {X_raw.shape} -> {X.shape} in {dt:.2f}s")

    save_embeddings(name, "NetLSD", X, y)
    print("Saved CSV+NPY. First row (first 5 dims):", np.round(X[0][:5], 4))
    print(f"\n=== NetLSD :: {name} ===")
    graphs, y = load_tu_dataset(name, DATASETS_ROOT)
    print(f"Loaded {len(graphs)} graphs; classes: {sorted(set(y.tolist()))}")

    t0 = time.time()
    # Instantiate without unsupported args (e.g., 'k')
    model = NetLSD(**{k: v for k, v in NETLSD_PARAMS.items() if v is not None})
    model.fit(graphs)
    X_raw = model.get_embedding()
    X = ensure_dimensionality(X_raw, TARGET_DIM)
    dt = time.time() - t0
    print(f"NetLSD produced raw shape {X_raw.shape} -> {X.shape} in {dt:.2f}s")

    save_embeddings(name, "NetLSD", X, y)
    print("Saved CSV+NPY. First row (first 5 dims):", np.round(X[0][:5], 4))

if __name__ == "__main__":
    print("DATASETS_ROOT:", DATASETS_ROOT)
    print("STRICT_LOCAL_ONLY:", STRICT_LOCAL_ONLY)
    for ds in DATASETS:
        try:
            run_netlsd_for_dataset(ds)
        except Exception as e:
            print(f"[WARN] {ds}: {e}")
