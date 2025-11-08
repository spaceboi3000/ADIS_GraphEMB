import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Paths
EMB_ROOT = Path("./embeddings/embeddings_netlsd")
OUT_FILE = Path("./scripts/netlsd/netlsd_classification_results.csv")
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
DIMS = ["dim64", "dim128", "dim256"]

results = []

for name in DATASETS:
    for dim in DIMS:
        csv_path = EMB_ROOT / name / dim / "NetLSD_embeddings.csv"
        if not csv_path.exists():
            print(csv_path)
            print(f"[WARN] Missing embeddings for {name} ({dim}), skipping.")
            continue

        print(f"\n=== Evaluating {name} | {dim} ===")
        df = pd.read_csv(csv_path)
        y = df["label"].values
        X = df.drop(columns=["label"]).values
        n_classes = len(set(y))
        print(f"Loaded {X.shape[0]} samples × {X.shape[1]} dims, {n_classes} classes.")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train classifier
        clf = SVC(kernel="rbf", probability=True)
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        if n_classes > 2:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        else:
            auc = roc_auc_score(y_test, y_prob[:, 1])

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f} | TrainT: {train_time:.2f}s")
        results.append(dict(
            Dataset=name,
            Dim=dim,
            Accuracy=acc,
            F1=f1,
            AUC=auc,
            TrainTime=train_time
        ))

# Save summary
res_df = pd.DataFrame(results)
res_df.to_csv(OUT_FILE, index=False)
print("\nSaved results →", OUT_FILE)
print(res_df)
