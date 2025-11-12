# /content/gin_embeddings_emb1.py
import argparse, time, csv, traceback, os, tracemalloc, math, json, random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from torch_geometric.nn import (
    GINConv,
    global_mean_pool, global_add_pool, global_max_pool,
    BatchNorm,
    JumpingKnowledge,
)
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, NormalizeFeatures
import torch_geometric.transforms as T


from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dropout_edge

import psutil


# Config 
# DATASETS_ROOT = Path("../../DATASETS")      
# DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
# OUT_DIR = Path("../../embeddings/embeddings_classification_gin")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
# SEED = 42


BASE = "./"
DATASETS_ROOT = f"{BASE}/DATASETS"
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
OUT_DIR = f"{BASE}/gin_embedding_classification"
SEED = 42

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
        if write_header and header:
            w.writerow(header)
        w.writerow(row)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    if probs.size == 0:
        return float('nan'), float('nan'), float('nan'), np.array([], dtype=int)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    classes = np.unique(y_true)
    try:
        if len(classes) == 1:
            auc_macro = float('nan')
        elif len(classes) == 2:
            pos_lbl = classes.max()
            pos_idx = int(pos_lbl)
            if pos_idx < probs.shape[1]:
                auc_macro = roc_auc_score(y_true, probs[:, pos_idx])
            else:
                auc_macro = float('nan')
        else:
            auc_macro = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
    except Exception:
        auc_macro = float('nan')
    return acc, f1_macro, auc_macro, y_pred


class DropEdgeTransform(BaseTransform):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def forward(self, data):
        if self.p <= 0.0 or data.edge_index is None:
            return data
        ei, mask = dropout_edge(data.edge_index, p=self.p, training=True)
        data.edge_index = ei
        # Keep edge attributes/weights in sync if they exist
        if getattr(data, 'edge_attr', None) is not None:
            data.edge_attr = data.edge_attr[mask]
        if getattr(data, 'edge_weight', None) is not None:
            data.edge_weight = data.edge_weight[mask]
        return data


class EnsureStrongX(BaseTransform):
    """
    If x is missing, create [one_hot_degree (<=max_degree), raw_degree] features.
    """
    def __init__(self, max_degree: int = 128):
        super().__init__()
        self.max_degree = max_degree

    def forward(self, data):  # <-- PyG expects forward()
        if getattr(data, 'x', None) is None:
            num_nodes = int(data.num_nodes)
            deg = degree(data.edge_index[0], num_nodes=num_nodes)
            clamped = torch.clamp(deg, max=self.max_degree).long()
            one_hot = F.one_hot(clamped, num_classes=self.max_degree + 1).float()
            data.x = torch.cat([one_hot, deg.view(-1, 1)], dim=-1)
        return data

def build_transform_chain(apply_norm: bool) -> T.Compose:  # optional: annotate as Compose
    tr_list = [EnsureStrongX(max_degree=128)]
    if apply_norm:
        tr_list.append(NormalizeFeatures())
    return T.Compose(tr_list)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

class GINBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.5, train_eps=True):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim, p=p)
        self.conv = GINConv(self.mlp, train_eps=train_eps)
        self.bn   = BatchNorm(out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        self.p = p

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.p, training=self.training)
        if self.res_proj is not None:
            x = self.res_proj(x)
        return out + x

def _graph_level_features(edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n_per_g = torch.bincount(batch)
    e_batch = batch[edge_index[0]]
    m_per_g = torch.bincount(e_batch, minlength=n_per_g.size(0))
    n = n_per_g.float()
    m = m_per_g.float()
    denom = torch.clamp(n * (n - 1), min=1.0)
    dens = (2.0 * m) / denom
    return torch.stack([n, m, dens], dim=1)

class GINGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=5, pooling="add", dropout=0.5, train_eps=True, jk_mode="cat"):
        super().__init__()
        self.dropout = dropout
        self.pooling = pooling
        self.layers = nn.ModuleList()
        self.layers.append(GINBlock(in_dim, hidden_dim, p=dropout, train_eps=train_eps))
        for _ in range(num_layers - 1):
            self.layers.append(GINBlock(hidden_dim, hidden_dim, p=dropout, train_eps=train_eps))
        self.jk = JumpingKnowledge(jk_mode)
        self.num_layers = num_layers

    def _pool(self, x, batch):
        if self.pooling == "add":
            return global_add_pool(x, batch)
        elif self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "concat":
            return torch.cat([
                global_add_pool(x, batch),
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=-1)
        else:
            return global_add_pool(x, batch)

    def forward(self, x, edge_index, batch):
        xs = []
        for layer in self.layers:
            x = layer(x, edge_index)
            xs.append(x)
        x = self.jk(xs)  # [N, hidden_dim * num_layers] if 'cat'
        g = self._pool(x, batch)
        return g

class GINForClassification(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers=5, pooling="add",
                 dropout=0.5, use_gfeat=False, jk_mode="cat", train_eps=True):
        super().__init__()
        self.use_gfeat = use_gfeat
        self.pooling = pooling
        self.encoder = GINGraphEncoder(in_dim, hidden_dim, num_layers, pooling, dropout, train_eps, jk_mode)
        enc_out = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        if pooling == "concat":
            enc_out *= 3
        if self.use_gfeat:
            enc_out += 3  # |V|,|E|,density
        self.classifier = nn.Linear(enc_out, num_classes)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        if self.use_gfeat:
            gfeat = _graph_level_features(data.edge_index, data.batch).to(z.device)
            z = torch.cat([z, gfeat], dim=-1)
        return self.classifier(z), z


class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -math.inf
        self.epochs_no_improve = 0
        self.best_state = None
    def step(self, val_acc: float, model: nn.Module):
        if val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.epochs_no_improve = 0
            self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.epochs_no_improve += 1
            return self.epochs_no_improve > self.patience

class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup:
            scale = epoch / max(1, self.warmup)
            return [base_lr * scale for base_lr in self.base_lrs]
        t = (epoch - self.warmup) / max(1, self.max_epochs - self.warmup)
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return [base_lr * cos for base_lr in self.base_lrs]

def train_epoch(model, loader, optimizer, device, scaler=None, grad_clip: float = 1.0,
                ce_weight=None, label_smooth: float = 0.0, dropedge_transform=None):
    model.train()
    total = 0.0
    for data in loader:
        # train-only DropEdge augmentation (apply on CPU tensors before .to(device))
        if dropedge_transform is not None:
            data = dropedge_transform(data)
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits, _ = model(data)
                loss = F.cross_entropy(logits, data.y.view(-1).long(),
                                       weight=ce_weight, label_smoothing=label_smooth)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(data)
            loss = F.cross_entropy(logits, data.y.view(-1).long(),
                                   weight=ce_weight, label_smoothing=label_smooth)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total += float(loss.item()) * data.num_graphs
    return total / max(len(loader.dataset), 1)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_embeds, all_labels, all_logits = [], [], []
    for data in loader:
        data = data.to(device)
        logits, z = model(data)
        preds = logits.argmax(dim=-1)
        correct += (preds == data.y.view(-1)).sum().item()
        total += data.num_graphs
        all_embeds.append(z.detach().cpu().numpy())
        all_labels.append(data.y.view(-1).detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())
    acc = correct / max(total, 1)
    embeds = np.concatenate(all_embeds, axis=0) if all_embeds else np.zeros((0,))
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,))
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,))
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy() if logits.size else np.zeros((0,))
    return acc, embeds, labels, logits, probs

def split_dataset(dataset, test_size=0.1, val_size=0.1, seed=42):
    y = np.array([int(d.y.item()) for d in dataset])
    idx = np.arange(len(dataset))
    try:
        idx_trainval, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
        y_trainval = y[idx_trainval]
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_size/(1.0-test_size), random_state=seed, stratify=y_trainval
        )
    except Exception:
        rng = np.random.RandomState(seed); rng.shuffle(idx)
        n_test = int(len(idx) * test_size); n_val = int(len(idx) * val_size)
        idx_test = idx[:n_test]; idx_val = idx[n_test:n_test+n_val]; idx_train = idx[n_test+n_val:]
    return dataset[idx_train.tolist()], dataset[idx_val.tolist()], dataset[idx_test.tolist()], idx_train, idx_val, idx_test

def save_embeddings(run_dir: Path, split_name: str, embeddings: np.ndarray, labels: np.ndarray, indices: np.ndarray):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / f"{split_name}_embeddings.npy", embeddings)
    with open(run_dir / f"{split_name}_labels.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["graph_index", "label"])
        for i, lbl in zip(indices, labels):
            w.writerow([int(i), int(lbl)])
    D = embeddings.shape[1] if embeddings.ndim == 2 else 0
    wide_csv = run_dir / f"{split_name}_embeddings_wide.csv"
    with open(wide_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["label"] + [f"dim{j}" for j in range(D)]
        w.writerow(header)
        for lbl, vec in zip(labels, embeddings):
            w.writerow([int(lbl)] + [float(x) for x in vec.tolist()])


def run(args):
    set_seed(SEED)
    device = torch.device(args.device)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    root = Path(args.data_root)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "metrics_summary.csv"

    summary_header = [
        "dataset","dim","epochs","lr","dropout",
        "it_time_s","embed_time_s","total_time_s",
        "train_acc","val_acc","test_acc",
        "train_f1_macro","val_f1_macro","test_f1_macro",
        "train_auc_macro","val_auc_macro","test_auc_macro",
        "peak_tracemalloc_mb","rss_before_mb","rss_after_mb"
    ]

    for ds_name in args.datasets:
        ds_dir = out_root / ds_name
        try:
            print(f"\n=== Dataset: {ds_name} ===")
            base_transform = build_transform_chain(apply_norm=args.norm_feats)
            dataset = TUDataset(root=str(root), name=ds_name,
                                transform=base_transform, use_node_attr=args.use_node_attr)

            num_classes = dataset.num_classes
            in_dim = dataset.num_features if dataset.num_features > 0 else dataset[0].x.size(1)

            train_dataset, val_dataset, test_dataset, idx_train, idx_val, idx_test = split_dataset(
                dataset, 0.1, 0.1, SEED
            )

            # loaders
            bs = args.batch_size
            train_loader_tr = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            train_loader_ev = DataLoader(train_dataset, batch_size=bs, shuffle=False)
            val_loader     = DataLoader(val_dataset,   batch_size=bs, shuffle=False)
            test_loader    = DataLoader(test_dataset,  batch_size=bs, shuffle=False)

            # class weights (from training subset)
            train_y = torch.tensor([int(d.y.item()) for d in train_dataset])
            class_counts = torch.bincount(train_y, minlength=num_classes)
            weights = (1.0 / (class_counts.float() + 1e-6))
            weights = (weights / weights.mean()).to(device)

            # train-time augmentation transform (DropEdge) applied per-batch in train_epoch
            # dropedge_transform = T.DropEdge(p=args.dropedge) if args.dropedge > 0.0 else None
            dropedge_transform = DropEdgeTransform(p=args.dropedge) if args.dropedge > 0.0 else None

            for dim in args.dims:
                for ep in args.epochs:
                    for lr in args.lrs:
                        for dropout in args.dropouts:
                            print(f"---> GIN | dim={dim}, epochs={ep}, lr={lr}, drop={dropout}, "
                                  f"pool={args.pool}, gfeat={args.gfeat}, norm={args.norm_feats}, "
                                  f"use_node_attr={args.use_node_attr}, jk=cat, train_eps=True, "
                                  f"ls={args.label_smoothing}, dropedge={args.dropedge}")
                            model = GINForClassification(
                                in_dim, dim, num_classes,
                                num_layers=args.num_layers, pooling=args.pool, dropout=dropout,
                                use_gfeat=args.gfeat, jk_mode="cat", train_eps=True
                            ).to(device)

                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
                            scheduler = WarmupCosine(optimizer, warmup_epochs=min(10, max(3, ep//10)), max_epochs=ep)

                            # run dir + curve log file
                            run_dir = ds_dir / f"gin_dim{dim}_ep{ep}_lr{lr}_drop{dropout}_pool{args.pool}_gfeat{int(args.gfeat)}_norm{int(args.norm_feats)}_attr{int(args.use_node_attr)}_ls{args.label_smoothing}_de{args.dropedge}"
                            run_dir.mkdir(parents=True, exist_ok=True)
                            curve_path = run_dir / "train_val_curve.csv"
                            _append_csv_row(curve_path, ["epoch","train_acc","val_acc","lr"], [0, "", "", ""])  # header

                            rss_before = _rss_mb()
                            tracemalloc.start()
                            t0 = time.time()

                            early = EarlyStopper(patience=args.patience, min_delta=1e-4)
                            best_val = -1.0

                            for epoch in range(1, ep + 1):
                                loss = train_epoch(model, train_loader_tr, optimizer, device,
                                                   scaler=scaler, grad_clip=args.grad_clip,
                                                   ce_weight=weights, label_smooth=args.label_smoothing,
                                                   dropedge_transform=dropedge_transform)
                                tr_acc, *_ = eval_epoch(model, train_loader_ev, device)
                                val_acc_cur, *_ = eval_epoch(model, val_loader, device)

                                # log curve
                                current_lr = optimizer.param_groups[0]['lr']
                                _append_csv_row(curve_path, None, [epoch, tr_acc, val_acc_cur, current_lr])

                                scheduler.step()

                                # early stop tracking
                                stop = early.step(val_acc_cur, model)
                                best_val = max(best_val, val_acc_cur)

                                if epoch % max(1, ep // 10) == 0 or epoch in (1, ep):
                                    print(f"  Epoch {epoch:03d} | loss={loss:.4f} | train_acc={tr_acc:.4f} | val_acc={val_acc_cur:.4f} | lr={current_lr:.6g}")

                                if stop:
                                    print(f"  Early stop at epoch {epoch} (best val_acc={early.best:.4f})")
                                    break

                            it_time_s = time.time() - t0

                            # Load best model by val acc
                            if early.best_state is not None:
                                model.load_state_dict({k: v.to(device) for k, v in early.best_state.items()})

                            # Embedding/generation timing
                            t_emb0 = time.time()
                            train_acc, train_emb, train_lbl, _, train_probs = eval_epoch(model, train_loader_ev, device)
                            val_acc,   val_emb,   val_lbl,   _, val_probs   = eval_epoch(model, val_loader,     device)
                            test_acc,  test_emb,  test_lbl,  _, test_probs  = eval_epoch(model, test_loader,    device)
                            embed_time_s = time.time() - t_emb0


                            # train_acc, train_emb, train_lbl = eval_epoch(model, train_loader, device)
                            # val_acc,   val_emb,   val_lbl   = eval_epoch(model, val_loader, device)
                            # test_acc,  test_emb,  test_lbl  = eval_epoch(model, test_loader, device)
                            
                            # train_acc, train_emb, train_lbl, train_logits, train_probs = eval_epoch(model, train_loader, device)
                            # val_acc,   val_emb,   val_lbl,   val_logits,   val_probs   = eval_epoch(model, val_loader, device)
                            # test_acc,  test_emb,  test_lbl,  test_logits,  test_probs  = eval_epoch(model, test_loader, device)

                            # Secondary metrics
                            test_acc2, test_f1, test_auc, test_pred = compute_metrics(test_lbl, test_probs)
                            val_acc2,  val_f1,  val_auc,  _        = compute_metrics(val_lbl,  val_probs)
                            train_acc2,train_f1,train_auc,_       = compute_metrics(train_lbl, train_probs)

                            test_acc = test_acc2
                            val_acc  = val_acc2
                            train_acc = train_acc2

                            current, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            peak_tracemalloc_mb = float(peak) / (1024.0**2)
                            rss_after = _rss_mb()
                            total_time_s = it_time_s + embed_time_s

                            
                            save_embeddings(run_dir, "train", train_emb, train_lbl, idx_train)
                            save_embeddings(run_dir, "val",   val_emb,   val_lbl,   idx_val)
                            save_embeddings(run_dir, "test",  test_emb,  test_lbl,  idx_test)

                            metrics = {
                                "dataset": ds_name,
                                "embedding_dim": dim, "epochs": ep, "lr": lr, "dropout": dropout,
                                "batch_size": args.batch_size, "num_layers": args.num_layers, "pooling": args.pool,
                                "train_time_sec": float(it_time_s),
                                "embed_time_sec": float(embed_time_s),
                                "total_time_sec": float(total_time_s),
                                "val_best_acc": float(early.best if early.best != -math.inf else best_val),
                                "train_acc": float(train_acc), "val_acc": float(val_acc), "test_acc": float(test_acc),
                                "train_f1_macro": float(train_f1), "val_f1_macro": float(val_f1), "test_f1_macro": float(test_f1),
                                "train_auc_macro": float(train_auc), "val_auc_macro": float(val_auc), "test_auc_macro": float(test_auc),
                                "peak_tracemalloc_mb": float(peak_tracemalloc_mb),
                                "rss_before_mb": float(rss_before), "rss_after_mb": float(rss_after),
                                "device": str(device),
                            }
                            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

                            cm = confusion_matrix(test_lbl, test_pred)
                            np.savetxt(run_dir / "test_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
                            (run_dir / "classification_report_test.txt").write_text(
                                classification_report(test_lbl, test_pred, digits=4)
                            )

                            row = [ds_name, dim, ep, lr, dropout,
                                   it_time_s, embed_time_s, total_time_s,
                                   train_acc,val_acc, test_acc, train_f1 ,val_f1, test_f1,train_auc, val_auc, test_auc,
                                   peak_tracemalloc_mb, rss_before, rss_after]
                            
                            # row = [ds_name, dim, ep, lr, dropout, it_time_s, embed_time_s, total_time_s, val_acc, test_acc, peak_tracemalloc_mb, rss_before, rss_after]
                            # row = [
                            #     ds_name, dim, ep, lr, dropout,
                            #     it_time_s, embed_time_s, total_time_s,
                            #     val_acc2, test_acc2,
                            #     val_f1, test_f1,
                            #     val_auc, test_auc,
                            #     peak_tracemalloc_mb, rss_before, rss_after
                            # ]
                            # _append_csv_row(run_csv, header, row)

                            # _append_csv_row(summary_csv, header, row)
                            # _append_csv_row(run_dir / "metrics.csv", perrun_header, row)
                            # _append_csv_row(summary_csv,          summary_header, row)

                            _append_csv_row(run_dir / "metrics.csv", summary_header, row)
                            _append_csv_row(summary_csv,          summary_header, row)

                            torch.save(model.state_dict(), run_dir / "gin_classifier.pt")
                            print(f"[Saved] {run_dir} | test_acc={test_acc:.4f} | it={it_time_s:.1f}s | emb={embed_time_s:.1f}s | peak={peak_tracemalloc_mb:.1f}MB")

        except Exception:
            err_dir = ds_dir / "_ERROR"; err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "traceback.txt").write_text(traceback.format_exc())
            print(f"[ERROR] {ds_name} failed. See {err_dir/'traceback.txt'}")



# Entry

# MUTAG epochs 500 // dropout 0.0, 0.4 // num_layers 5 // weight_decay 5e-4 //batch_size 64
# ENZYMES epochs 500 // dropout 0.0, 0.5 // num_layers 6 // weight_decay 5e-4 //batch_size 32 
# IMDB-MULTI epochs 200 // dropout 0.0, 0.4 // num_layers 3 // weight_decay 1e-4 //batch_size 64

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GIN embeddings exporter for EMB1 (strong features + JK + residuals + label smoothing + DropEdge)")
    p.add_argument("--data_root", type=str, default=str(DATASETS_ROOT))
    p.add_argument("--out_root", type=str, default=str(OUT_DIR))
    p.add_argument("--datasets", nargs="+", default=['MUTAG'])
    p.add_argument("--dims", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--epochs", nargs="+", type=int, default=[200, 500])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 5e-4])
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.4])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--pool", type=str, default="add", choices=["mean", "add", "concat"])
    p.add_argument("--gfeat", action="store_true", help="Concatenate graph-level features |V|,|E|,density")
    p.add_argument("--norm_feats", action="store_true", help="Apply NormalizeFeatures() after EnsureStrongX()")
    p.add_argument("--use_node_attr", action="store_true", help="Use continuous node attributes if available (e.g., ENZYMES)")
    p.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing for CE")
    p.add_argument("--dropedge", type=float, default=0.2, help="DropEdge probability (train batches only)")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = p.parse_args()
    run(args)
