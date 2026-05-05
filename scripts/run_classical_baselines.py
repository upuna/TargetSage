#!/usr/bin/env python3
"""
Run classical ML baselines (LR, SVM, KNN, NB, RF, GB, MLP) on bio-only features.
Matches the evaluation protocol of train.py: 80/20 stratified split, Adjusted F1, 5 seeds.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from targetsage import (
    TASKS, TASK_DISPLAY,
    load_bio_features, load_labels,
)
from targetsage.metrics import adjusted_f1


def make_classifiers():
    return {
        "LR":  LogisticRegression(max_iter=5000, C=1.0),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=4),
        "NB":  GaussianNB(),
        "RF":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4),
        "GB":  GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        "XGB": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             eval_metric="logloss", use_label_encoder=False,
                             random_state=42, n_jobs=4),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio",    default="data/gene_features.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", default="results/classical_baselines")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[DATA] Loading bio features...")
    bio_df  = load_bio_features(args.bio)
    label_df = load_labels(args.labels)

    merged = label_df.merge(bio_df, on="Gene_Symbol", how="inner")
    bio_cols = [c for c in merged.columns if c not in ["Gene_Symbol"] + TASKS]

    Xb_raw = merged[bio_cols].to_numpy(dtype=float)
    Xb = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(Xb_raw)
    )
    print(f"  {len(merged)} genes, {Xb.shape[1]} bio features")

    all_results = []

    for task in TASKS:
        if task not in merged.columns:
            continue
        y_all = merged[task].to_numpy(dtype=float)
        valid = ~np.isnan(y_all)
        if valid.sum() < 50:
            continue

        display = TASK_DISPLAY.get(task, task)
        y = y_all[valid].astype(int)
        X = Xb[valid]
        n_pos = int(y.sum())
        if n_pos < 5:
            continue

        print(f"\n[{display}] |P|={n_pos} |U|={len(y)-n_pos}")

        for clf_name, clf_template in make_classifiers().items():
            seed_scores = []
            for seed in args.seeds:
                np.random.seed(seed)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed
                )
                try:
                    from sklearn.base import clone
                    clf = clone(clf_template)
                    # Handle class imbalance for tree methods
                    if clf_name in ("RF", "GB"):
                        pass  # sklearn handles via criterion
                    if clf_name == "XGB":
                        n_neg = (y_tr == 0).sum()
                        clf.set_params(scale_pos_weight=n_neg / max(y_tr.sum(), 1))
                    clf.fit(X_tr, y_tr)
                    probs = clf.predict_proba(X_te)[:, 1]
                    af1 = adjusted_f1(y_te, probs)
                    seed_scores.append(af1)
                    print(f"  {clf_name} seed={seed}: adj_f1={af1:.4f}")
                except Exception as e:
                    print(f"  {clf_name} seed={seed}: ERROR {e}")

            if seed_scores:
                mean_s = float(np.mean(seed_scores))
                std_s  = float(np.std(seed_scores))
                all_results.append({
                    "task": task, "display": display, "method": clf_name,
                    "adj_f1_mean": mean_s, "adj_f1_std": std_s, "n_seeds": len(seed_scores),
                })
                print(f"  {clf_name}: {mean_s:.2f} ± {std_s:.2f}")

    df = pd.DataFrame(all_results)
    out_path = os.path.join(args.outdir, "classical_baselines_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")

    # Print pivot table
    pivot = df.pivot_table(index="display", columns="method", values="adj_f1_mean")
    print("\nSummary Adjusted F1 (%):")
    print(pivot.round(1).to_string())
    macros = df.groupby("method")["adj_f1_mean"].mean()
    print("\nMacro averages:")
    print(macros.round(1).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
