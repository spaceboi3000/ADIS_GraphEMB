import torch
import random
import os
from pathlib import Path
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops

# === USER SETTINGS ===
DATASETS_ROOT = Path("../DATASETS")  
OUTPUT_ROOT = Path("../permutated_DATASETS")
OUTPUT_ROOT.mkdir(exist_ok=True)
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]

PERTURB_EDGE_PERCENT = 0.10    # 10% edges removed + 10% added
SHUFFLE_NODE_ATTR = True       # random permutation of node features


def perturb_graph(data, perturb_ratio=0.10, shuffle_attr=False):
    
    edge_index = data.edge_index.clone()

    #Remove Edge
    num_edges = edge_index.size(1)
    num_remove = int(perturb_ratio * num_edges)

    if num_remove > 0:
        perm = torch.randperm(num_edges)[:num_remove]
        edge_index = edge_index[:, perm]  # keep only random subset

    #Add Random Edges
    num_add = num_remove
    if num_add > 0:
        nodes = data.num_nodes
        new_edges = torch.randint(0, nodes, (2, num_add))  # random pairs
        edge_index = torch.cat([edge_index, new_edges], dim=1)

    #Shuffle Node Attributes
    if shuffle_attr and data.x is not None:
        perm_nodes = torch.randperm(data.x.size(0))
        data.x = data.x[perm_nodes]

    #clean graph
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    edge_index = to_undirected(edge_index)

    data.edge_index = edge_index
    
    return data

for ds_name in DATASETS:
    
    print(f"\nLoading dataset: {ds_name}")
    
    ds_path = DATASETS_ROOT / ds_name
    perm_path = OUTPUT_ROOT / ds_name
    perm_path.mkdir(exist_ok=True)

    dataset = TUDataset(root=ds_path, name=ds_name)

    permutated_list = []

    for i, data in enumerate(dataset):
        perturbed = perturb_graph(
            data, 
            perturb_ratio=PERTURB_EDGE_PERCENT,
            shuffle_attr=SHUFFLE_NODE_ATTR
        )
        permutated_list.append(perturbed)

    torch.save(permutated_list, perm_path / f"{ds_name}_permutated.pt")
    print(f"Saved: {perm_path}/{ds_name}_permutated.pt")

