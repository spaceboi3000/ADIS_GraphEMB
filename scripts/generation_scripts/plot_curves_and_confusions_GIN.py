
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def safe_read_csv(path, **kwargs):
    """Read CSV safely; return None if failed."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def plot_training_curve(run_dir: Path, curve_csv: Path):
    """Plot training and validation accuracy vs epochs."""
    df = safe_read_csv(curve_csv)
    if df is None or df.empty:
        return None

    # Clean and convert numeric
    df = df.rename(columns={"epoch": "epoch", "train_acc": "train_acc", "val_acc": "val_acc", "lr": "lr"})
    df = df[pd.to_numeric(df["epoch"], errors="coerce").notna()].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype(int)
    for col in ["train_acc", "val_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Plot
    fig = plt.figure(figsize=(15,10))
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training Curve: {run_dir.name}")
    plt.legend()
    out_path = run_dir / "training_curve.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")
    return out_path


def plot_confusion_matrix(run_dir: Path, cm_csv: Path):
    """Plot confusion matrix heatmap."""
    try:
        cm = np.loadtxt(cm_csv, delimiter=",").astype(int)
    except Exception:
        return None

    fig = plt.figure(figsize=(15,10))
    plt.imshow(cm, aspect="auto", cmap="viridis")
    plt.title(f"Confusion Matrix: {run_dir.name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    out_path = run_dir / "confusion_matrix.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")
    return out_path


def main(out_root: str):
    OUT_ROOT = Path(out_root)
    if not OUT_ROOT.exists():
        print(f"[ERROR] Output directory not found: {OUT_ROOT}")
        return

    dataset_dirs = [Path(p) for p in glob(str(OUT_ROOT / "*")) if Path(p).is_dir()]
    run_dirs = []
    for ds in dataset_dirs:
        for p in glob(str(ds / "gin_dim*")):
            if Path(p).is_dir():
                run_dirs.append(Path(p))

    if not run_dirs:
        print("[INFO] No run directories found.")
        return

    print(f"[INFO] Found {len(run_dirs)} runs under {OUT_ROOT}")

    for rd in run_dirs:
        curve_csv = rd / "train_val_curve.csv"
        if curve_csv.exists():
            plot_training_curve(rd, curve_csv)

        cm_csv = rd / "test_confusion_matrix.csv"
        if cm_csv.exists():
            plot_confusion_matrix(rd, cm_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="./gin_embedding_classification")
    args = ap.parse_args()
    main(args.out_root)
