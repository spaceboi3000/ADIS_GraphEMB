import argparse, time, csv, traceback, os, tracemalloc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, BatchNorm
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
import psutil
import json



#config
DATASETS_ROOT = Path("../DATASETS")      
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]

OUT_DIR = Path("./embeddings_gin")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def set_seed(seed: int = 42):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _rss_mb():
    try:
        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss) / (1024.0**2)
    except Exception:
        return float('nan')

def _append_csv_row(csv_path: Path, header: list, row: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


#transform-always ensure x
class EnsureX(BaseTransform): 

    def forward(self, data):
        if getattr(data, 'x', None) is None:
            num_nodes = int(data.num_nodes)
            if data.edge_index is not None and data.edge_index.numel() > 0:
                deg = degree(data.edge_index[0], num_nodes=num_nodes).unsqueeze(1)
            else:
                deg = torch.zeros((num_nodes, 1), dtype=torch.float)
            if deg.sum() == 0:
                deg = torch.ones((num_nodes, 1), dtype=torch.float)
            data.x = deg
        return data

#model
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class GINGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=5, pooling="mean", dropout=0.0):
        super().__init__()
        self.dropout = dropout; self.pooling = pooling
        self.layers = nn.ModuleList(); self.norms = nn.ModuleList()
        self.layers.append(GINConv(MLP(in_dim, hidden_dim)))
        self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(MLP(hidden_dim, hidden_dim)))
            self.norms.append(BatchNorm(hidden_dim))
        self.proj_head = nn.Identity()
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.layers, self.norms):
            x = conv(x, edge_index)
            x = bn(x)        
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch) if self.pooling == "mean" else global_add_pool(x, batch)
        return self.proj_head(g)

class GINForClassification(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=5, pooling="mean", dropout=0.0):
        super().__init__()
        self.encoder = GINGraphEncoder(in_dim, hidden_dim, num_layers, pooling, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        return self.classifier(z), z

#train and valll
def train_epoch(model, loader, optimizer, device):

    model.train(); total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits, _ = model(data)
        loss = F.cross_entropy(logits, data.y.view(-1).long())
        loss.backward(); optimizer.step()
        total += loss.item() * data.num_graphs

    return total / max(len(loader.dataset), 1)



@torch.no_grad()
def eval_epoch(model, loader, device):

    model.eval(); correct = 0; total = 0
    all_embeds, all_labels = [], []

    for data in loader:
        data = data.to(device)
        logits, z = model(data)
        preds = logits.argmax(dim=-1)
        correct += (preds == data.y.view(-1)).sum().item()
        total += data.num_graphs
        all_embeds.append(z.cpu().numpy())
        all_labels.append(data.y.view(-1).cpu().numpy())

    acc = correct / max(total, 1)
    embeds = np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0,))
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))
    return acc, embeds, labels


def split_dataset(dataset, test_size=0.1, val_size=0.1, seed=42):

    y = np.array([d.y.item() for d in dataset]); idx = np.arange(len(dataset))
    try:
        idx_trainval, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
        y_trainval = y[idx_trainval]
        idx_train, idx_val = train_test_split(idx_trainval, test_size=val_size/(1.0-test_size), random_state=seed, stratify=y_trainval)
    except Exception:
        rng = np.random.RandomState(seed); rng.shuffle(idx)
        n_test = int(len(idx)*test_size); n_val = int(len(idx)*val_size)
        idx_test = idx[:n_test]; idx_val = idx[n_test:n_test+n_val]; idx_train = idx[n_test+n_val:]
    return dataset[idx_train.tolist()], dataset[idx_val.tolist()], dataset[idx_test.tolist()], idx_train, idx_val, idx_test



def save_embeddings(run_dir: Path, split_name: str, embeddings: np.ndarray, labels: np.ndarray, indices: np.ndarray):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"{split_name}_embeddings.npy", embeddings)    # npy
    with open(run_dir / f"{split_name}_labels.csv", "w", newline="") as f:   # mapping CSV
        w = csv.writer(f); w.writerow(["graph_index","label"])
        for i,lbl in zip(indices, labels): w.writerow([int(i), int(lbl)])
  
    D = embeddings.shape[1] if embeddings.ndim == 2 else 0   # wide CSV: label, dim0..dim{D-1}
    wide_csv = run_dir / f"{split_name}_embeddings_wide.csv"
    with open(wide_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["label"] + [f"dim{j}" for j in range(D)]
        w.writerow(header)
        for lbl, vec in zip(labels, embeddings):
            w.writerow([int(lbl)] + [float(x) for x in vec.tolist()])


def run(args):
    set_seed(42)
    device = torch.device(args.device)
    root = Path(args.data_root)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "metrics_summary.csv"
    header = ["dataset","dim","epochs","lr","dropout","it_time_s","embed_time_s","total_time_s","val_acc","test_acc","peak_tracemalloc_mb","rss_before_mb","rss_after_mb"]

    for ds_name in args.datasets:
        ds_dir = out_root / ds_name
        try:
            print(f"\n=== Dataset: {ds_name} ===")
            dataset = TUDataset(root=str(root), name=ds_name, transform=EnsureX())
            num_classes = dataset.num_classes
            in_dim = dataset.num_features if (dataset.num_features and dataset.num_features > 0) else 1
            train_dataset, val_dataset, test_dataset, idx_train, idx_val, idx_test = split_dataset(dataset, 0.1, 0.1, 42)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

            for dim in args.dims:
                for ep in args.epochs:
                    for lr in args.lrs:
                        for dropout in args.dropouts:
                            print(f"---> GIN | dim={dim}, epochs={ep}, lr={lr}, dropout={dropout}")
                            model = GINForClassification(in_dim, dim, num_classes, args.num_layers, args.pool, dropout).to(device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                            #timings and memory
                            rss_before = _rss_mb()
                            tracemalloc.start()
                            t0 = time.time()

                            best_val, best_state = -1.0, None
                            for epoch in range(1, ep+1):
                                loss = train_epoch(model, train_loader, optimizer, device)
                                val_acc_cur, _, _ = eval_epoch(model, val_loader, device)
                                if val_acc_cur > best_val:
                                    best_val = val_acc_cur
                                    best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                                if epoch % 10 == 0 or epoch in (1, ep):
                                    print(f"  Epoch {epoch:03d} | loss={loss:.4f} | val_acc={val_acc_cur:.4f}")

                            it_time_s = time.time() - t0

                            if best_state:
                                model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

                            #embedding/generation timing
                            t_emb0 = time.time()
                            train_acc, train_emb, train_lbl = eval_epoch(model, train_loader, device)
                            val_acc,   val_emb,   val_lbl   = eval_epoch(model, val_loader, device)
                            test_acc,  test_emb,  test_lbl  = eval_epoch(model, test_loader, device)
                            embed_time_s = time.time() - t_emb0

                            current, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            peak_tracemalloc_mb = float(peak) / (1024.0**2)
                            rss_after = _rss_mb()
                            total_time_s = it_time_s + embed_time_s

                            run_dir = ds_dir / f"gin_dim{dim}_ep{ep}_lr{lr}_drop{dropout}"
                            save_embeddings(run_dir, "train", train_emb, train_lbl, idx_train)
                            save_embeddings(run_dir, "val",   val_emb,   val_lbl,   idx_val)
                            save_embeddings(run_dir, "test",  test_emb,  test_lbl,  idx_test)

        
                            metrics = {
                                "dataset": ds_name, "embedding_dim": dim, "epochs": ep, "lr": lr, "dropout": dropout,
                                "batch_size": args.batch_size, "num_layers": args.num_layers, "pooling": args.pool,
                                "train_time_sec": float(it_time_s),
                                "embed_time_sec": float(embed_time_s),
                                "total_time_sec": float(total_time_s),
                                "val_best_acc": float(best_val),
                                "train_acc": float(train_acc), "val_acc": float(val_acc), "test_acc": float(test_acc),
                                "peak_tracemalloc_mb": float(peak_tracemalloc_mb),
                                "rss_before_mb": float(rss_before), "rss_after_mb": float(rss_after),
                                "device": str(device),
                            }
                            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    
                            run_csv = run_dir / "metrics.csv"
                            row = [ds_name, dim, ep, lr, dropout, it_time_s, embed_time_s, total_time_s, val_acc, test_acc, peak_tracemalloc_mb, rss_before, rss_after]
                            _append_csv_row(run_csv, header, row)

                            _append_csv_row(summary_csv, header, row)

                            torch.save(model.state_dict(), run_dir / "gin_classifier.pt")
                            print(f"[Saved] {run_dir} | test_acc={test_acc:.4f} | it={it_time_s:.1f}s | emb={embed_time_s:.1f}s | peak={peak_tracemalloc_mb:.1f}MB")

        except Exception:
            err_dir = ds_dir / "_ERROR"; err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "traceback.txt").write_text(traceback.format_exc())
            print(f"[ERROR] {ds_name} failed. See {err_dir/'traceback.txt'}")

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="GIN embeddings exporter for EMB1")
    
    p.add_argument("--data_root", type=str, default=str(DATASETS_ROOT))
    p.add_argument("--out_root", type=str, default=str(OUT_DIR))
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--dims", nargs="+", type=int, default=[64,128,256])
    p.add_argument("--epochs", nargs="+", type=int, default=[40,100])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-3,5e-4])
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.0,0.3])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--pool", type=str, default="mean", choices=["mean","add"])
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = p.parse_args([]) 
    print("Running GIN embeddings exporter with:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    set_seed(SEED)
    run(args)


