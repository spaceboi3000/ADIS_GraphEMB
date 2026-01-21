from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import numpy as np

DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]

GIN_ORIG_ROOT = Path("../embeddings/embeddings_gin")
GIN_PERM_ROOT = Path("../permutated_embeddings/permutated_gin_embedding_classification")

OUT_CSV = Path("../permutated_embeddings/stability_gin.csv")

def rowwise_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity per row between A and B."""
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch in rowwise_cosine: A{A.shape} vs B{B.shape}")
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    A_norm = np.nan_to_num(A_norm)
    B_norm = np.nan_to_num(B_norm)
    return np.sum(A_norm * B_norm, axis=1)


def load_gin_embeddings_and_labels(run_dir: Path, split: str = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings + labels CSV for given run_dir and split (train/val/test).

    Returns:
      E: (N, D) embeddings
      graph_index: (N,) int array
      labels: (N,) int array
    """
    emb_path = run_dir / f"{split}_embeddings.npy"
    lab_path = run_dir / f"{split}_labels.csv"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    if not lab_path.exists():
        raise FileNotFoundError(f"Labels not found: {lab_path}")

    E = np.load(emb_path)

    graph_indices: List[int] = []
    labels: List[int] = []

    with open(lab_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "graph_index" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"'graph_index' or 'label' column missing in {lab_path}")
        for row in reader:
            graph_indices.append(int(row["graph_index"]))
            labels.append(int(row["label"]))

    if E.shape[0] != len(graph_indices):
        raise ValueError(
            f"Row mismatch: {emb_path} has {E.shape[0]} rows, {lab_path} has {len(graph_indices)} rows"
        )

    return E, np.array(graph_indices, dtype=int), np.array(labels, dtype=int)


def compute_gin_stability_for_run(
    dataset: str,
    run_name: str,
    split: str = "test",
) -> Dict:
    """
    Compare original vs permutated GIN embeddings for a specific dataset + run directory.
    Align by graph_index using the labels CSVs.
    """
    run_orig = GIN_ORIG_ROOT / dataset / run_name
    run_perm = GIN_PERM_ROOT / dataset / run_name

    if not run_orig.exists():
        raise FileNotFoundError(f"Original run dir not found: {run_orig}")
    if not run_perm.exists():
        raise FileNotFoundError(f"Permutated run dir not found: {run_perm}")

    E_orig, gidx_orig, labels_orig = load_gin_embeddings_and_labels(run_orig, split=split)
    E_perm, gidx_perm, labels_perm = load_gin_embeddings_and_labels(run_perm, split=split)

    # Build mapping from graph_index -> row_idx
    map_orig = {int(g): i for i, g in enumerate(gidx_orig)}
    map_perm = {int(g): i for i, g in enumerate(gidx_perm)}

    # intersection of graph indices
    common_g = sorted(set(map_orig.keys()) & set(map_perm.keys()))

    if not common_g:
        raise ValueError(f"No overlapping graph_index between original and permutated runs for {dataset}/{run_name}")

    row_orig = np.array([map_orig[g] for g in common_g], dtype=int)
    row_perm = np.array([map_perm[g] for g in common_g], dtype=int)

    Z_orig = E_orig[row_orig]
    Z_perm = E_perm[row_perm]

    # Extra debug: print dimensions if mismatch
    if Z_orig.shape[1] != Z_perm.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch for {dataset}/{run_name}: "
            f"orig D={Z_orig.shape[1]}, perm D={Z_perm.shape[1]}"
        )

    cos = rowwise_cosine(Z_orig, Z_perm)
    l2 = np.linalg.norm(Z_orig - Z_perm, axis=1)

    return {
        "dataset": dataset,
        "run_name": run_name,
        "split": split,
        "n_graphs": int(Z_orig.shape[0]),
        "cos_mean": float(cos.mean()),
        "cos_median": float(np.median(cos)),
        "cos_std": float(cos.std()),
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_std": float(l2.std()),
    }


if __name__ == "__main__":
    rows: List[Dict] = []

    for ds in DATASETS:
        orig_ds_root = GIN_ORIG_ROOT / ds
        perm_ds_root = GIN_PERM_ROOT / ds

        if not orig_ds_root.exists():
            print(f"[WARN] Original GIN dir for dataset {ds} not found: {orig_ds_root}")
            continue
        if not perm_ds_root.exists():
            print(f"[WARN] Permutated GIN dir for dataset {ds} not found: {perm_ds_root}")
            continue

        # find common run directories (same hyperparam configs)
        orig_runs = {p.name for p in orig_ds_root.iterdir() if p.is_dir() and not p.name.startswith("_")}
        perm_runs = {p.name for p in perm_ds_root.iterdir() if p.is_dir() and not p.name.startswith("_")}
        common_runs = sorted(orig_runs.intersection(perm_runs))

        if not common_runs:
            print(f"[WARN] No common run dirs for dataset {ds}")
            continue

        for run_name in common_runs:
            try:
                res = compute_gin_stability_for_run(dataset=ds, run_name=run_name, split="test")
                rows.append(res)
                print(f"[OK] {ds} | {run_name} | n={res['n_graphs']} | cos_mean={res['cos_mean']:.4f}")
            except Exception as e:
                print(f"[WARN] {ds} | {run_name}: {repr(e)}")

    if rows:
        # write CSV manually (no pandas)
        fieldnames = list(rows[0].keys())
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[Saved] GIN stability metrics -> {OUT_CSV}")
    else:
        print("\n[ERROR] No GIN stability metrics computed.")
