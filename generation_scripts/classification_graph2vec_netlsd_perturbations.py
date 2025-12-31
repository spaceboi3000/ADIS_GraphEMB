import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


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

def get_balanced_classifiers(n_samples, n_classes, n_features):
    """Return classifiers with balanced hyperparameters to prevent overfitting"""
    
    classifiers = {}
    
    # SVM 
    c_param = 10.0 if n_samples > 150 else 1.0
    classifiers["SVM"] = SVC(
        kernel="rbf", 
        C=c_param,
        gamma="scale", 
        probability=True, 
        random_state=42
    )
    
    # Logistic Regression
    classifiers["LogisticRegression"] = LogisticRegression(
        C=5.0 if n_samples < 200 else 1.0,
        max_iter=2000,
        random_state=42,
        solver="lbfgs",
        multi_class="multinomial" if n_classes > 2 else "auto"
    )
    
    # Random Forest
    if n_samples < 150:
        max_depth = 8
        n_trees = 500
    else:
        max_depth = 20
        n_trees = 300
    
    classifiers["RandomForest"] = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    )
    
    # Gradient Boosting 
    classifiers["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=200 if n_samples > 150 else 150,
        learning_rate=0.1,
        max_depth=5 if n_samples > 150 else 4,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    
    # MLP 
    if n_features > 200:
        hidden = (256, 128, 64)
    elif n_features > 100:
        hidden = (128, 64)
    else:
        hidden = (64, 32)
    
    classifiers["MLP"] = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=1000,
        learning_rate_init=0.001,
        alpha=0.005, 
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42
    )
    
    return classifiers


#a voting ensemble of best performing models
def create_ensemble(n_samples):
    
    #adaptive ensemble based on dataset size
    if n_samples < 150:
        rf = RandomForestClassifier(
            n_estimators=500, 
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=300, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    svm = SVC(
        kernel="rbf",
        C=10.0 if n_samples > 150 else 1.0,
        probability=True,
        random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble

def evaluate_embeddings(method_name: str, emb_root: Path, out_file: Path):
    results = []
    
    for name in DATASETS:
        for dim in DIMS:
            csv_path = emb_root / name / dim / f"{method_name}_embeddings.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing embeddings for {name} ({dim}) in {method_name}, skipping.")
                continue
            
            print(f"\n{'='*70}")
            print(f"Evaluating {method_name} | {name} | {dim}")
            print('='*70)
            
            df = pd.read_csv(csv_path)
            y = df["label"].values
            X = df.drop(columns=["label"]).values
            n_classes = len(set(y))
            n_samples = X.shape[0]
            n_features = X.shape[1]
            
            print(f"Samples: {n_samples} | Features: {n_features} | Classes: {n_classes}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Get balanced classifiers
            classifiers = get_balanced_classifiers(n_samples, n_classes, X.shape[1])
            
            # Add ensemble model
            try:
                ensemble = create_ensemble(n_samples)
                classifiers["Ensemble"] = ensemble
            except:
                pass
            
            # Track best model
            best_acc = 0
            best_model = ""
            
            # Evaluate each classifier
            for clf_name, clf in classifiers.items():
                try:
                    print(f"  {clf_name:20s}", end=" ")
                    metrics = evaluate_model(clf_name, clf, X_train_scaled, X_test_scaled, 
                                            y_train, y_test, n_classes)
                    
                    acc_str = f"Acc: {metrics['Accuracy']:.4f}"
                    f1_str = f"F1: {metrics['F1']:.4f}"
                    auc_str = f"AUC: {metrics['AUC']:.4f}"
                    time_str = f"Time: {metrics['TrainTime']:.2f}s"
                    
                    print(f"| {acc_str} | {f1_str} | {auc_str} | {time_str}")
                    
                    if metrics['Accuracy'] > best_acc:
                        best_acc = metrics['Accuracy']
                        best_model = clf_name
                    
                    results.append(dict(
                        Method=method_name,
                        Dataset=name,
                        Dim=dim,
                        Model=metrics["Model"],
                        Accuracy=metrics["Accuracy"],
                        F1=metrics["F1"],
                        AUC=metrics["AUC"],
                        TrainTime=metrics["TrainTime"]
                    ))
                except Exception as e:
                    print(f"ERROR: {str(e)}")
                    continue
            
            print(f"\n  ★ Best Model: {best_model} with Accuracy: {best_acc:.4f}")
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {out_file}")
    print('='*70)
    
    # Summary by classifier
    print("\n" + "="*70)
    print("AVERAGE PERFORMANCE BY CLASSIFIER")
    print("="*70)
    summary = res_df.groupby("Model")[["Accuracy", "F1", "AUC"]].agg(["mean", "std"])
    print(summary.round(4))
    
    # Summary by dataset
    print("\n" + "="*70)
    print("AVERAGE PERFORMANCE BY DATASET")
    print("="*70)
    dataset_summary = res_df.groupby("Dataset")[["Accuracy", "F1", "AUC"]].agg(["mean", "max"])
    print(dataset_summary.round(4))
    
    # Best configurations
    print("\n" + "="*70)
    print("BEST CONFIGURATION PER DATASET")
    print("="*70)
    best_per_dataset = res_df.loc[res_df.groupby("Dataset")["Accuracy"].idxmax()]
    print(best_per_dataset[["Dataset", "Dim", "Model", "Accuracy", "F1", "AUC"]].to_string(index=False))
    
    return res_df


print("\n" + "="*70)
print("STARTING IMPROVED EVALUATION PIPELINE")
print("="*70)

netlsd_results = evaluate_embeddings(
    method_name="NetLSD",
    emb_root=Path("../permutated_embeddings/permutated_netlsd"),
    out_file=Path("../csvs/perturbated_netlsd_classification_results.csv"),
)

graph2vec_results = evaluate_embeddings(
    method_name="Graph2Vec",
    emb_root=Path("../permutated_embeddings/permutated_graph2vec"),
    out_file=Path("../csvs/perturbated_graph2vec_classification_results.csv"),
)

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)

# Compare methods
print("\n" + "="*70)
print("COMPARISON: NetLSD vs Graph2Vec")
print("="*70)
comparison = pd.DataFrame({
    'NetLSD_Acc': netlsd_results.groupby('Dataset')['Accuracy'].mean(),
    'Graph2Vec_Acc': graph2vec_results.groupby('Dataset')['Accuracy'].mean(),
    'NetLSD_F1': netlsd_results.groupby('Dataset')['F1'].mean(),
    'Graph2Vec_F1': graph2vec_results.groupby('Dataset')['F1'].mean()
})
comparison['Winner'] = comparison.apply(
    lambda x: 'NetLSD' if x['NetLSD_Acc'] > x['Graph2Vec_Acc'] else 'Graph2Vec', axis=1
)
print(comparison.round(4))