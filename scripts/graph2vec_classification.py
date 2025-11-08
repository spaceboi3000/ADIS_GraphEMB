import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Paths
EMB_ROOT = Path("./embeddings_graph2vec")
OUT_FILE = Path("./scripts/graph2vec_classification_results.csv")
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]

# Embedding dimensions to test
EMBEDDING_DIMS = [32, 64, 128, 256]

# Experiment configuration
N_RUNS = 5  # Number of repetitions for each configuration
RANDOM_SEEDS = [42, 123, 456, 789, 999]  # Different seeds for each run

results = []

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

for name in DATASETS:
    csv_path = EMB_ROOT / name / "Graph2Vec_embeddings.csv"
    if not csv_path.exists():
        print(f"[WARN] Missing embeddings for {name}, skipping.")
        continue
    
    print(f"\n{'='*60}")
    print(f"=== Evaluating {name} ===")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    y = df["label"].values
    X_full = df.drop(columns=["label"]).values
    n_classes = len(set(y))
    original_dim = X_full.shape[1]
    
    print(f"Loaded {X_full.shape[0]} samples × {original_dim} dims, {n_classes} classes.")
    print(f"Running {N_RUNS} repetitions with different random splits...")
    
    # Determine which dimensions to test
    dims_to_test = [d for d in EMBEDDING_DIMS if d <= original_dim]
    if original_dim not in dims_to_test:
        dims_to_test.append(original_dim)
    dims_to_test = sorted(set(dims_to_test))
    
    print(f"Testing embedding dimensions: {dims_to_test}")
    
    # Define classifiers
    classifiers = {
        "SVM": lambda seed: SVC(kernel="rbf", probability=True, random_state=seed, max_iter=1000),
        "MLP": lambda seed: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, 
                                         random_state=seed, early_stopping=True)
    }
    
    # Test different embedding dimensions
    for emb_dim in dims_to_test:
        print(f"\n--- Testing dimension: {emb_dim} ---")
        
        # Multiple runs with different random splits
        for run_idx, seed in enumerate(RANDOM_SEEDS[:N_RUNS]):
            print(f"\n  Run {run_idx + 1}/{N_RUNS} (seed: {seed})")
            
            # Train/test split (stratified) - different split each run
            X_train_full, X_test_full, y_train, y_test = train_test_split(
                X_full, y, test_size=0.2, random_state=seed, stratify=y
            )
            
            # Truncate or use full embeddings
            if emb_dim < original_dim:
                X_train = X_train_full[:, :emb_dim]
                X_test = X_test_full[:, :emb_dim]
            else:
                X_train = X_train_full
                X_test = X_test_full
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Record embedding generation time (approximate as 0 since pre-computed)
            generation_time = 0.0
            
            for clf_name, clf_fn in classifiers.items():
                print(f"    Training {clf_name}...")
                
                # Initialize classifier with current seed
                clf = clf_fn(seed)
                
                # Memory before training
                mem_before = get_memory_usage()
                
                # Training
                start = time.time()
                clf.fit(X_train_scaled, y_train)
                train_time = time.time() - start
                
                # Memory after training
                mem_after = get_memory_usage()
                memory_used = mem_after - mem_before
                
                # Predictions
                pred_start = time.time()
                y_pred = clf.predict(X_test_scaled)
                y_prob = clf.predict_proba(X_test_scaled)
                pred_time = time.time() - pred_start
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                
                # AUC calculation
                try:
                    if n_classes > 2:
                        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                    else:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                except Exception as e:
                    print(f"    Warning: Could not compute AUC - {e}")
                    auc = np.nan
                
                print(f"    {clf_name:4s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
                print(f"          | Train: {train_time:.3f}s | Pred: {pred_time:.3f}s | Mem: {memory_used:.2f}MB")
                
                # Save results with run identifier
                results.append({
                    "Dataset": name,
                    "Classifier": clf_name,
                    "Embedding_Dim": emb_dim,
                    "Run": run_idx + 1,
                    "Random_Seed": seed,
                    "N_Samples": X_train.shape[0] + X_test.shape[0],
                    "N_Classes": n_classes,
                    "Accuracy": round(acc, 4),
                    "F1_Score": round(f1, 4),
                    "AUC": round(auc, 4) if not np.isnan(auc) else None,
                    "Train_Time_sec": round(train_time, 3),
                    "Prediction_Time_sec": round(pred_time, 3),
                    "Generation_Time_sec": generation_time,
                    "Memory_MB": round(memory_used, 2),
                    "Train_Samples": X_train.shape[0],
                    "Test_Samples": X_test.shape[0]
                })

# Save detailed results
print(f"\n{'='*60}")
print("Saving results...")
res_df = pd.DataFrame(results)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
res_df.to_csv(OUT_FILE, index=False)
print(f"Results saved → {OUT_FILE}")

# Create summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS (Mean ± Std across runs)")
print(f"{'='*60}\n")

# Calculate mean and standard deviation for each configuration
summary_stats = res_df.groupby(['Dataset', 'Classifier', 'Embedding_Dim']).agg({
    'Accuracy': ['mean', 'std'],
    'F1_Score': ['mean', 'std'], 
    'AUC': ['mean', 'std'],
    'Train_Time_sec': ['mean', 'std'],
    'Memory_MB': ['mean', 'std']
}).round(4)

# Format the summary nicely
for (dataset, classifier, emb_dim), group_data in res_df.groupby(['Dataset', 'Classifier', 'Embedding_Dim']):
    acc_mean = group_data['Accuracy'].mean()
    acc_std = group_data['Accuracy'].std()
    f1_mean = group_data['F1_Score'].mean()
    f1_std = group_data['F1_Score'].std()
    auc_mean = group_data['AUC'].mean()
    auc_std = group_data['AUC'].std()
    
    print(f"{dataset} | {classifier} (dim={emb_dim}):")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
    print()

# Best results per dataset (considering mean performance)
print(f"\n{'='*60}")
print("BEST CONFIGURATIONS PER DATASET (Mean Accuracy)")
print(f"{'='*60}\n")

for dataset in res_df['Dataset'].unique():
    dataset_df = res_df[res_df['Dataset'] == dataset]
    
    # Calculate mean accuracy for each configuration
    mean_accuracies = dataset_df.groupby(['Classifier', 'Embedding_Dim'])['Accuracy'].mean()
    best_config = mean_accuracies.idxmax()
    best_mean_acc = mean_accuracies.max()
    
    # Get all runs for the best configuration
    best_runs = dataset_df[
        (dataset_df['Classifier'] == best_config[0]) & 
        (dataset_df['Embedding_Dim'] == best_config[1])
    ]
    
    acc_mean = best_runs['Accuracy'].mean()
    acc_std = best_runs['Accuracy'].std()
    f1_mean = best_runs['F1_Score'].mean()
    f1_std = best_runs['F1_Score'].std()
    auc_mean = best_runs['AUC'].mean()
    auc_std = best_runs['AUC'].std()
    
    print(f"{dataset}:")
    print(f"  Best: {best_config[0]} (dim={best_config[1]})")
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1-Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  AUC:      {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  Across {len(best_runs)} runs\n")

# Display full results table
print(f"\nDetailed results table (first 20 rows):\n")
print(res_df.head(20).to_string(index=False))

print(f"\nTotal experiments: {len(res_df)}")
print(f"Configurations: {len(res_df[['Dataset', 'Classifier', 'Embedding_Dim']].drop_duplicates())}")
print(f"Runs per configuration: {N_RUNS}")