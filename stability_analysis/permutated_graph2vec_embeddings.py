from __future__ import annotations
import time
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import networkx as nx

from karateclub import Graph2Vec

import torch
from torch_geometric.utils import to_networkx

#original dataset
DATASETS_ROOT = Path("../DATASETS")

#PERMUTATED datasets 
PERMUTATED_ROOT = Path("../permutated_DATASETS")

DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
EMBEDDING_DIMS = [64, 128, 256]

OUT_DIR = Path("../permutated_embeddings/permutated_graph2vec")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = OUT_DIR / "permutated_metrics.csv"



def load_permutated_dataset(name: str, perm_root: Path) -> Tuple[List[nx.Graph], np.ndarray]:
    """
    Load a permutated dataset previously saved.
    Each element is a PyG Data object with perturbed edges and shuffled node features.
    Converts each Data to a NetworkX graph with node feature 'feat' (if x exists).
    Returns: list of NetworkX graphs, and numpy array of labels.
    """
    perm_path = perm_root / name / f"{name}_permutated.pt"
    if not perm_path.exists():
        raise FileNotFoundError(f"Permutated dataset file not found: {perm_path}")

    data_list = torch.load(perm_path)
    graphs: List[nx.Graph] = []
    labels = []

    for data in data_list:
        
        #convert to undirected NetworkX graph
        g = to_networkx(data, to_undirected=True)

        #attach node features if they exist
        if getattr(data, "x", None) is not None:
            x_np = data.x.cpu().numpy()
            
            # ensure node ordering is consistent
            for i, (_, d) in enumerate(g.nodes(data=True)):
                d["feat"] = x_np[i].tolist()

        graphs.append(g)
        labels.append(int(data.y.item()))

    return graphs, np.array(labels)


def make_graph2vec_params(dim: int) -> Dict:
    """Return a consistent set of Graph2Vec params for a given dimensionality."""
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



def run_graph2vec_for_permutated_dataset(name: str):
    print(f"\nPermutated Graph2Vec :: {name}")

    graphs, y = load_permutated_dataset(name, PERMUTATED_ROOT)
    print(f"Loaded {len(graphs)} permutated graphs; classes: {sorted(set(y.tolist()))}")

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
        peak_mb = peak / (1024 ** 2)
        rss_before_mb = rss_before / (1024 ** 2)
        rss_after_mb = rss_after / (1024 ** 2)

        print(f"dim={dim} -> shape {X.shape}, total {total_time:.2f}s, peak_mem {peak_mb:.1f} MB")

        # Save embeddings for this dataset/dim
        save_embeddings(name, "Graph2Vec", dim, X, y)

        metrics_rows.append({
            "dataset": name,
            "method": "Graph2Vec_permutated",
            "dim": dim,
            "n_graphs": len(graphs),
            "fit_time_s": round(fit_time, 4),
            "embed_time_s": round(embed_time, 4),
            "total_time_s": round(total_time, 4),
            "rss_before_mb": round(rss_before_mb, 2),
            "rss_after_mb": round(rss_after_mb, 2),
            "peak_tracemalloc_mb": round(peak_mb, 2),
        })

    mdf = pd.DataFrame(metrics_rows)
    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        mdf = pd.concat([old, mdf], ignore_index=True)
    mdf.to_csv(METRICS_PATH, index=False)
    print(f"Metrics appended to {METRICS_PATH}")


if __name__ == "__main__":
    print("PERMUTATED_ROOT:", PERMUTATED_ROOT)
    print("OUT_DIR:", OUT_DIR)
    for ds in DATASETS:
        try:
            run_graph2vec_for_permutated_dataset(ds)
        except Exception as e:
            print(f"[WARN] {ds}: {e}")
