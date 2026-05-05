#!/usr/bin/env python3
"""
Run additional baselines: XGBoost, TabNet, ESM-2+MLP
Uses same data splits and Adjusted F1 metric as train.py.
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from targetsage import (
    TASKS, TASK_DISPLAY,
    load_bio_features, load_llm_scores, load_llm_embeddings,
    load_labels, build_feature_matrix, get_feature_arrays,
)
from targetsage.metrics import adjusted_f1


def run_xgboost(X_train, y_train, X_test):
    from xgboost import XGBClassifier
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric='logloss', use_label_encoder=False,
        random_state=42, n_jobs=4,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def run_tabnet(X_train, y_train, X_test):
    from pytorch_tabnet.tab_model import TabNetClassifier
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5,
        gamma=1.5, lambda_sparse=1e-3,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=__import__('torch').optim.lr_scheduler.StepLR,
        verbose=0, seed=42,
    )
    # TabNet needs float32
    X_tr = X_train.astype(np.float32)
    X_te = X_test.astype(np.float32)
    y_tr = y_train.astype(int)

    # Simple train/val split for TabNet
    from sklearn.model_selection import train_test_split
    X_t, X_v, y_t, y_v = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    clf.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        eval_metric=['logloss'],
        max_epochs=100, patience=15, batch_size=256,
        weights=1, drop_last=False,
    )
    return clf.predict_proba(X_te)[:, 1]


def run_esm2_mlp(X_train_emb, y_train, X_test_emb):
    """Simple MLP on ESM-2 embeddings (simulating PLM-only baseline)."""
    import torch
    import torch.nn as nn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simple 2-layer MLP
    d_in = X_train_emb.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 1),
    ).to(device)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    X_t = torch.from_numpy(X_train_emb.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    X_te = torch.from_numpy(X_test_emb.astype(np.float32)).to(device)

    model.train()
    for epoch in range(100):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 256):
            idx = perm[i:i+256]
            logits = model(X_t[idx]).squeeze()
            loss = criterion(logits, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te).squeeze()).cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio", default="data/gene_features.tsv")
    ap.add_argument("--scores", default="data/features_llm_structured_scores.csv")
    ap.add_argument("--embeddings", default="data/features_llm_embedding.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", default="results/new_baselines")
    ap.add_argument("--emb_pca_dim", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[DATA] Loading...")
    bio_df = load_bio_features(args.bio)
    score_df = load_llm_scores(args.scores)
    emb_df = load_llm_embeddings(args.embeddings)
    label_df = load_labels(args.labels)

    merged = build_feature_matrix(bio_df, score_df, emb_df, label_df)
    Xb, Xa, Xe, genes = get_feature_arrays(merged)

    # Preprocess
    Xb = SimpleImputer(strategy="median").fit_transform(Xb)
    Xa = SimpleImputer(strategy="median").fit_transform(Xa)
    Xe = SimpleImputer(strategy="median").fit_transform(Xe)

    if args.emb_pca_dim > 0 and Xe.shape[1] > args.emb_pca_dim:
        Xe = PCA(n_components=args.emb_pca_dim, random_state=42).fit_transform(Xe)

    # Concat all features for XGBoost and TabNet
    X_all = np.hstack([
        StandardScaler().fit_transform(Xb),
        StandardScaler().fit_transform(Xa),
        StandardScaler().fit_transform(Xe),
    ])

    # ESM-2 embeddings only (simulate PLM baseline)
    X_emb = StandardScaler().fit_transform(Xe)

    print(f"  X_all: {X_all.shape}, X_emb: {X_emb.shape}")

    # Methods to run
    methods = {
        "XGBoost": lambda Xtr, ytr, Xte: run_xgboost(Xtr, ytr, Xte),
        "TabNet": lambda Xtr, ytr, Xte: run_tabnet(Xtr, ytr, Xte),
    }

    esm_method = {
        "ESM2+MLP": lambda Xtr, ytr, Xte: run_esm2_mlp(Xtr, ytr, Xte),
    }

    all_results = []

    for task in TASKS:
        if task not in merged.columns:
            continue
        y_all = merged[task].to_numpy()
        valid = ~np.isnan(y_all)
        if valid.sum() < 50:
            continue

        display = TASK_DISPLAY.get(task, task)
        y = y_all[valid].astype(int)
        X_task = X_all[valid]
        X_emb_task = X_emb[valid]

        n_pos = int(y.sum())
        if n_pos < 5:
            continue

        print(f"\n[{display}] |P|={n_pos} |U|={len(y)-n_pos}")

        for method_name, method_fn in methods.items():
            seed_scores = []
            for seed in args.seeds:
                np.random.seed(seed)
                # 80/20 stratified split
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_task, y, test_size=0.2, stratify=y, random_state=seed
                )
                try:
                    probs = method_fn(X_tr, y_tr, X_te)
                    af1 = adjusted_f1(y_te, probs)
                    seed_scores.append(af1)
                    print(f"  {method_name} seed={seed}: adj_f1={af1:.4f}")
                except Exception as e:
                    print(f"  {method_name} seed={seed}: ERROR {e}")

            if seed_scores:
                mean_s = np.mean(seed_scores)
                std_s = np.std(seed_scores)
                all_results.append({
                    "task": task, "display": display, "method": method_name,
                    "adj_f1_mean": mean_s, "adj_f1_std": std_s, "n_seeds": len(seed_scores),
                })
                print(f"  {method_name}: {mean_s:.2f}±{std_s:.2f}")

        # ESM-2 + MLP (uses embedding features only)
        for method_name, method_fn in esm_method.items():
            seed_scores = []
            for seed in args.seeds:
                np.random.seed(seed)
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_emb_task, y, test_size=0.2, stratify=y, random_state=seed
                )
                try:
                    probs = method_fn(X_tr, y_tr, X_te)
                    af1 = adjusted_f1(y_te, probs)
                    seed_scores.append(af1)
                    print(f"  {method_name} seed={seed}: adj_f1={af1:.4f}")
                except Exception as e:
                    print(f"  {method_name} seed={seed}: ERROR {e}")

            if seed_scores:
                mean_s = np.mean(seed_scores)
                std_s = np.std(seed_scores)
                all_results.append({
                    "task": task, "display": display, "method": method_name,
                    "adj_f1_mean": mean_s, "adj_f1_std": std_s, "n_seeds": len(seed_scores),
                })
                print(f"  {method_name}: {mean_s:.2f}±{std_s:.2f}")

    # Save
    df = pd.DataFrame(all_results)
    out_path = os.path.join(args.outdir, "new_baselines_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[DONE] Saved to {out_path}")

    # Print summary table
    pivot = df.pivot_table(index="display", columns="method", values="adj_f1_mean")
    print("\nSummary (Adjusted F1 %):")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
