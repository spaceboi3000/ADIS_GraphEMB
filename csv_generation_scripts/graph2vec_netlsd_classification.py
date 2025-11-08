import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Shared config
DATASETS = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
DIMS = ["dim64", "dim128", "dim256"]

def evaluate_model(model_name, clf, X_train, X_test, y_train, y_test, n_classes):
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        y_prob = clf.decision_function(X_test)
        if y_prob.ndim == 1:
            y_prob = np.vstack([1 - y_prob, y_prob]).T

    if n_classes > 2:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    else:
        auc = roc_auc_score(y_test, y_prob[:, 1])

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return dict(Accuracy=acc, F1=f1, AUC=auc, TrainTime=train_time, Model=model_name)

def evaluate_embeddings(method_name: str, emb_root: Path, out_file: Path):
    results = []
    for name in DATASETS:
        for dim in DIMS:
            csv_path = emb_root / name / dim / f"{method_name}_embeddings.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing embeddings for {name} ({dim}) in {method_name}, skipping.")
                continue

            print(f"\n=== Evaluating {method_name} | {name} | {dim} ===")
            df = pd.read_csv(csv_path)
            y = df["label"].values
            X = df.drop(columns=["label"]).values
            n_classes = len(set(y))
            print(f"Loaded {X.shape[0]} samples × {X.shape[1]} dims, {n_classes} classes.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # --- Classifier 1: SVM ---
            svm = SVC(kernel="rbf", probability=True)
            metrics_svm = evaluate_model("SVM", svm, X_train, X_test, y_train, y_test, n_classes)

            # --- Classifier 2: MLP ---
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                solver="adam", max_iter=500, random_state=42)
            metrics_mlp = evaluate_model("MLP", mlp, X_train, X_test, y_train, y_test, n_classes)

            for m in [metrics_svm, metrics_mlp]:
                results.append(dict(
                    Method=method_name,
                    Dataset=name,
                    Dim=dim,
                    Model=m["Model"],
                    Accuracy=m["Accuracy"],
                    F1=m["F1"],
                    AUC=m["AUC"],
                    TrainTime=m["TrainTime"]
                ))

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_file, index=False)
    print(f"\nSaved results → {out_file}")
    print(res_df)

# === RUN FOR BOTH METHODS ===

evaluate_embeddings(
    method_name="NetLSD",
    emb_root=Path("./embeddings/embeddings_netlsd"),
    out_file=Path("./csvs/netlsd_classification_results.csv"),
)

evaluate_embeddings(
    method_name="Graph2Vec",
    emb_root=Path("./embeddings/embeddings_graph2vec"),
    out_file=Path("./csvs/graph2vec_classification_results.csv"),
)
