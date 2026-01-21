from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
DIMS = [64, 128, 256]

# Graph2Vec paths
G2V_ORIG_ROOT = Path("../embeddings/embeddings_graph2vec")
G2V_PERM_ROOT = Path("../permutated_embeddings/permutated_graph2vec")

# NetLSD paths
NLSD_ORIG_ROOT = Path("../embeddings/embeddings_netlsd")
NLSD_PERM_ROOT = Path("../permutated_embeddings/permutated_netlsd")

OUT_CSV = Path("../permutated_embeddings/stability_unsupervised.csv")



def rowwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity per row between A and B."""
    assert A.shape == B.shape
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    A_norm = np.nan_to_num(A_norm)
    B_norm = np.nan_to_num(B_norm)
    return np.sum(A_norm * B_norm, axis=1)


def load_embeddings_pair(orig_path: Path,perm_path: Path,) -> (np.ndarray, np.ndarray):
    if not orig_path.exists():
        raise FileNotFoundError(f"Original embeddings not found: {orig_path}")
    if not perm_path.exists():
        raise FileNotFoundError(f"Permutated embeddings not found: {perm_path}")

    X_orig = np.load(orig_path)
    X_perm = np.load(perm_path)

    if X_orig.shape != X_perm.shape:
        raise ValueError(
            f"Shape mismatch: {orig_path} {X_orig.shape} vs {perm_path} {X_perm.shape}"
        )
    return X_orig, X_perm


def compute_stability(
    method: str,
    dataset: str,
    dim: int,
    orig_root: Path,
    perm_root: Path,
) -> Dict:
    """
    Compute cosine + L2 stability for a given method/dataset/dim.
    """
    if method == "Graph2Vec":
        fname = "Graph2Vec_embeddings.npy"
    elif method == "NetLSD":
        fname = "NetLSD_embeddings.npy"
    else:
        raise ValueError(f"Unknown method: {method}")

    orig_path = orig_root / dataset / f"dim{dim}" / fname
    perm_path = perm_root / dataset / f"dim{dim}" / fname

    X_orig, X_perm = load_embeddings_pair(orig_path, perm_path)

    cos = rowwise_cosine(X_orig, X_perm)
    l2 = np.linalg.norm(X_orig - X_perm, axis=1)

    return {
        "method": method,
        "dataset": dataset,
        "dim": dim,
        "n_graphs": X_orig.shape[0],
        # cosine stats
        "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "cos_std": float(cos.std()),
        # L2 stats
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()),
    }


if __name__ == "__main__":
    rows: List[Dict] = []

    for ds in DATASETS:
        for d in DIMS:
            # Graph2Vec
            try:
                r_g2v = compute_stability(
                    method="Graph2Vec",
                    dataset=ds,
                    dim=d,
                    orig_root=G2V_ORIG_ROOT,
                    perm_root=G2V_PERM_ROOT,
                )
                rows.append(r_g2v)
                print(f"[OK] Graph2Vec | {ds} | dim={d} | cos_mean={r_g2v['cos_mean']:.4f}")
            except Exception as e:
                print(f"[WARN] Graph2Vec | {ds} | dim={d}: {e}")

            # NetLSD
            try:
                r_nlsd = compute_stability(
                    method="NetLSD",
                    dataset=ds,
                    dim=d,
                    orig_root=NLSD_ORIG_ROOT,
                    perm_root=NLSD_PERM_ROOT,
                )
                rows.append(r_nlsd)
                print(f"[OK] NetLSD    | {ds} | dim={d} | cos_mean={r_nlsd['cos_mean']:.4f}")
            except Exception as e:
                print(f"[WARN] NetLSD    | {ds} | dim={d}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print(f"\n[Saved] Unsupervised stability metrics -> {OUT_CSV}")
    else:
        print("\n[ERROR] No stability metrics computed.")
