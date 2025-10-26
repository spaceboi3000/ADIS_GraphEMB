# filename: karate_script_demo.py

from karateclub import Graph2Vec
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1. Load MUTAG from TUDataset (PyTorch Geometric)
print("Loading MUTAG dataset...")
dataset = TUDataset(root="./DATASETS", name="MUTAG")

# Convert each PyG Data object to a networkx graph
graphs = [to_networkx(data, to_undirected=True) for data in dataset]
labels = dataset.data.y.tolist()

print(f"Loaded {len(graphs)} graphs with {len(set(labels))} classes.")

# 2. Train Graph2Vec
model = Graph2Vec(dimensions=128, workers=4, epochs=50)
print("Training Graph2Vec embeddings...")
model.fit(graphs)

# 3. Extract embeddings
embeddings = model.get_embedding()
print(f"Embeddings shape: {embeddings.shape}")

# 4. Visualize with t-SNE
X_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=40)
plt.title("Graph2Vec Embeddings (MUTAG)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Graph Class")
plt.show()
