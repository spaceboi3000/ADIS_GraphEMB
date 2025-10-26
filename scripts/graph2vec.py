from __future__ import annotations
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

import networkx as nx
from karateclub import Graph2Vec
from sklearn.utils import check_random_state

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# Config
DATASETS_ROOT = Path("../DATASETS")  
STRICT_LOCAL_ONLY = False           
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"] 
TARGET_DIM = 128
OUT_DIR = Path("./embeddings_graph2vec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH2VEC_PARAMS: Dict = dict(
    dimensions=TARGET_DIM,
    wl_iterations=2,
    min_count=5,
    learning_rate=0.025,
    epochs=15,
    seed=42,
)

# helpers
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

def save_embeddings(dataset: str, method: str, X: np.ndarray, y: np.ndarray):
    ds_dir = OUT_DIR / dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / f"{method}_embeddings.npy", X)
    df = pd.DataFrame(X)
    df.insert(0, "label", y)
    df.to_csv(ds_dir / f"{method}_embeddings.csv", index=False)


# main
def run_graph2vec_for_dataset(name: str):
    print(f"\n=== Graph2Vec :: {name} ===")
    graphs, y = load_tu_dataset(name, DATASETS_ROOT)
    print(f"Loaded {len(graphs)} graphs; classes: {sorted(set(y.tolist()))}")

    t0 = time.time()
    model = Graph2Vec(**GRAPH2VEC_PARAMS)
    model.fit(graphs)
    X = model.get_embedding()
    dt = time.time() - t0
    print(f"Graph2Vec produced shape {X.shape} in {dt:.2f}s")

    save_embeddings(name, "Graph2Vec", X, y)
    print("Saved CSV+NPY. First row (first 5 dims):", np.round(X[0][:5], 4))

if __name__ == "__main__":
    print("DATASETS_ROOT:", DATASETS_ROOT)
    print("STRICT_LOCAL_ONLY:", STRICT_LOCAL_ONLY)
    for ds in DATASETS:
        try:
            run_graph2vec_for_dataset(ds)
        except Exception as e:
            print(f"[WARN] {ds}: {e}")
