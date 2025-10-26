from __future__ import annotations
import time, os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

import networkx as nx
from karateclub import Graph2Vec

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# Config
DATASETS_ROOT = Path("../DATASETS")  
STRICT_LOCAL_ONLY = False             
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
EMBEDDING_DIMS = [64, 128, 256]    # different dim
OUT_DIR = Path("./embeddings_graph2vec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def make_graph2vec_params(dim: int) -> Dict:
    return dict(
        dimensions=dim,
        wl_iterations=2,
        min_count=5,
        learning_rate=0.025,
        epochs=15,
        seed=42,
    )

def save_embeddings(dataset: str, method: str, dim: int, X: np.ndarray, y: np.ndarray):
    ds_dir = OUT_DIR / dataset / f"dim{dim}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / f"{method}_embeddings.npy", X.astype(np.float32))
    cols = [f"dim{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "label", y.astype(int))
    df.to_csv(ds_dir / f"{method}_embeddings.csv", index=False)

# Main
def run_graph2vec_for_dataset(name: str):
    print(f"\n=== Graph2Vec :: {name} ===")
    graphs, y = load_tu_dataset(name, DATASETS_ROOT)
    print(f"Loaded {len(graphs)} graphs; classes: {sorted(set(y.tolist()))}")

    import tracemalloc, psutil
    proc = psutil.Process(os.getpid())

    metrics_rows = []
    for dim in EMBEDDING_DIMS:
        params = make_graph2vec_params(dim)

        tracemalloc.start()
        rss_before = proc.memory_info().rss
        t0 = time.time()
        model = Graph2Vec(**params)
        model.fit(graphs)
        fit_end = time.time()
        X = model.get_embedding()
        t1 = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = proc.memory_info().rss

        fit_time = fit_end - t0
        embed_time = t1 - fit_end
        total_time = t1 - t0
        peak_mb = peak / (1024**2)
        rss_before_mb = rss_before / (1024**2)
        rss_after_mb = rss_after / (1024**2)

        print(f"dim={dim} -> shape {X.shape}, total {total_time:.2f}s, peak_mem {peak_mb:.1f} MB")
        save_embeddings(name, "Graph2Vec", dim, X, y)
        metrics_rows.append({
            "dataset": name,
            "method": "Graph2Vec",
            "dim": dim,
            "n_graphs": len(graphs),
            "fit_time_s": round(fit_time, 4),
            "embed_time_s": round(embed_time, 4),
            "total_time_s": round(total_time, 4),
            "rss_before_mb": round(rss_before_mb, 2),
            "rss_after_mb": round(rss_after_mb, 2),
            "peak_tracemalloc_mb": round(peak_mb, 2),
        })

    # Append metrics CSV
    mpath = OUT_DIR / "metrics_graph2vec.csv"
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
            run_graph2vec_for_dataset(ds)
        except Exception as e:
            print(f"[WARN] {ds}: {e}")
