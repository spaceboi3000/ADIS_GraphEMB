from __future__ import annotations
import os, time, tracemalloc, itertools
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedShuffleSplit


from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant



# Config 
DATASETS_ROOT = Path("../DATASETS")      
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
TARGET_DIMS = [64, 128, 256]                 #embedding sizes to compare

#extra paramters for comparison
BATCH_SIZES = [32, 64]
EPOCHS_LIST = [40, 100]
LRS = [1e-3, 5e-4]
WEIGHT_DECAYS = [0.0]

OUT_DIR = Path("./embeddings_gin")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def set_seed(seed: int = SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# model 
class GINNet(nn.Module):

    def __init__(self, in_channels: int, hidden: int, out_dim: int, num_classes: int, num_layers: int = 5):
        super().__init__()
        def mlp_block(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Linear(out_c, out_c),
            )
        self.convs = nn.ModuleList()
        last_c = in_channels
        for _ in range(num_layers):
            self.convs.append(GINConv(mlp_block(last_c, hidden)))
            last_c = hidden
        self.proj = nn.Linear(hidden, out_dim)        #projection to target embedding dim
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index, batch):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)                #[num_graphs, hidden]
        z = self.proj(x)                              #graph embedding [num_graphs, out_dim]
        logits = self.classifier(z)
        return logits, z


#Training 
def load_dataset(name: str) -> TUDataset:

    ds = TUDataset(root=str(DATASETS_ROOT), name=name)
    if ds.num_node_features == 0:  #if dataset has no nodes attributes.l IMDB-MULTI has not, so we give  a constant feature
        ds = TUDataset(root=str(DATASETS_ROOT), name=name, transform=Constant(1.0))
    return ds


def make_splits(y: np.ndarray, seed: int = SEED):       # 80/10/10 stratified

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, tmp_idx = next(sss1.split(np.zeros_like(y), y))
    y_tmp = y[tmp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(sss2.split(np.zeros_like(y_tmp), y_tmp))
    val_idx = tmp_idx[val_idx]
    test_idx = tmp_idx[test_idx]
    return train_idx, val_idx, test_idx


def train_one_config(name: str, dim: int, batch_size: int, epochs: int, lr: float, weight_decay: float, device):
    
    set_seed(SEED)
    ds = load_dataset(name)

    # DOES NOT WORK FOR IMDB-MULTI
    # if ds.num_node_features == 0:
    #     ds.transform = Constant(value=1.0)
    #     # ds = load_dataset(name) 

    # labels = ds.data.y.cpu().numpy()
    labels = np.array([int(d.y) for d in ds])
    num_classes = int(labels.max() + 1)
    train_idx, val_idx, test_idx = make_splits(labels)

    train_loader = DataLoader(ds[train_idx.tolist()], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds[val_idx.tolist()], batch_size=batch_size)
    test_loader = DataLoader(ds[test_idx.tolist()], batch_size=batch_size)
    all_loader = DataLoader(ds, batch_size=batch_size)

    model = GINNet(in_channels=max(ds.num_node_features, 1), hidden=dim, out_dim=dim, num_classes=num_classes).to(device)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #measure training time and memory 
    proc = psutil.Process(os.getpid())
    tracemalloc.start()
    t0 = time.time()

    best_val = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y)
            opt.zero_grad(); loss.backward(); opt.step()

        #val accuracy
        model.eval(); correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x.float(), batch.edge_index, batch.batch)
                pred = logits.argmax(dim=-1)
                correct += int((pred == batch.y).sum().item())
                total += batch.y.size(0)
        val_acc = correct / max(total, 1)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    fit_time = time.time() - t0
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    peak_mb = peak / (1024**2)

    #restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    #test accuracy 
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch.x.float(), batch.edge_index, batch.batch)
            pred = logits.argmax(dim=-1)
            correct += int((pred == batch.y).sum().item())
            total += batch.y.size(0)
    test_acc = correct / max(total, 1)

    #the section where we generate embeddings for all graphs
    rss_before = proc.memory_info().rss
    t_emb0 = time.time()
    Z_list = []
    Y_list = []
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            _, z = model(batch.x.float(), batch.edge_index, batch.batch)
            Z_list.append(z.cpu())
            Y_list.append(batch.y.cpu())
    Z = torch.cat(Z_list, dim=0).numpy().astype(np.float32)
    Y = torch.cat(Y_list, dim=0).numpy().astype(int)
    embed_time = time.time() - t_emb0
    rss_after = proc.memory_info().rss

    total_time = fit_time + embed_time
    rss_before_mb = rss_before / (1024**2)
    rss_after_mb = rss_after / (1024**2)


    ds_dir = OUT_DIR / name / f"dim{dim}_bs{batch_size}_ep{epochs}_lr{lr:g}_wd{weight_decay:g}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    np.save(ds_dir / "GIN_embeddings.npy", Z)
    df = pd.DataFrame(Z, columns=[f"dim{i}" for i in range(Z.shape[1])])
    df.insert(0, "label", Y)
    df.to_csv(ds_dir / "GIN_embeddings.csv", index=False)

    return {
        "dataset": name,
        "method": "GIN",
        "dim": dim,
        "n_graphs": len(ds),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "fit_time_s": round(fit_time, 4),
        "embed_time_s": round(embed_time, 4),
        "total_time_s": round(total_time, 4),
        "val_acc": round(best_val, 4),
        "test_acc": round(test_acc, 4),
        "peak_tracemalloc_mb": round(peak_mb, 2),
        "rss_before_mb": round(rss_before_mb, 2),
        "rss_after_mb": round(rss_after_mb, 2),
        "device": str(device),
    }


def main():

    set_seed(SEED)
    device = get_device()
    # print("Device:", device)
    metrics_rows = []
    for name in DATASETS:
        for dim, bs, epochs, lr, wd in itertools.product(TARGET_DIMS, BATCH_SIZES, EPOCHS_LIST, LRS, WEIGHT_DECAYS):
            print(f"\n=== GIN :: {name} :: dim={dim} bs={bs} ep={epochs} lr={lr:g} wd={wd:g} ===")
            try:
                row = train_one_config(name, dim, bs, epochs, lr, wd, device)
                metrics_rows.append(row)
            except Exception as e:
                print(f"[WARN] {name} dim={dim} bs={bs} ep={epochs} lr={lr:g} wd={wd:g}: {e}")

    mpath = OUT_DIR / "metrics_gin.csv"
    mdf = pd.DataFrame(metrics_rows)
    if mpath.exists():
        old = pd.read_csv(mpath)
        mdf = pd.concat([old, mdf], ignore_index=True)
    mdf.to_csv(mpath, index=False)
    print(f"\nMetrics appended to {mpath}")

if __name__ == "__main__":
    main()
