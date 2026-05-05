#!/usr/bin/env python3
"""Re-run TabNet on High-Conf and SM(Appr) using official pytorch-tabnet library."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)

TASKS = [
    "task_triage_tier12_vs_others",    # High-Confidence
    "task_sm_bucket1_vs_others",       # SM (Appr.)
]
DISPLAY = {
    "task_triage_tier12_vs_others": "High-Confidence Targets",
    "task_sm_bucket1_vs_others": "Small-Molecule (Approved)",
}

def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0: return 0.0
    return float((probs[pos].mean() ** 2) / max(probs.mean(), 1e-10))

print("[DATA] loading...")
bio = pd.read_csv("data/gene_features.tsv", sep="\t").set_index("Gene_Symbol")
bio = bio.apply(pd.to_numeric, errors='coerce').fillna(0.0)
labels = pd.read_csv("data/gene_labels.tsv", sep="\t").set_index("Gene_Symbol")
genes = sorted(set(bio.index) & set(labels.index))
X_all = np.nan_to_num(bio.reindex(genes).values.astype(float), nan=0.0)
print(f"  {len(genes)} genes, X shape: {X_all.shape}")

from pytorch_tabnet.tab_model import TabNetClassifier

results = []
for task in TASKS:
    if task not in labels.columns:
        print(f"Task {task} not found in labels!")
        continue
    y_all = labels[task].reindex(genes).values.astype(float)
    valid = ~np.isnan(y_all)
    X = X_all[valid]; y = y_all[valid].astype(int)
    n_seeds = 5
    print(f"\n[{DISPLAY[task]}] |P|={y.sum()}, n={valid.sum()}, n_seeds={n_seeds}")
    scores = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr).astype(np.float32)
        X_te_s = sc.transform(X_te).astype(np.float32)

        clf = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=None,
            verbose=0, seed=seed,
        )
        # Compute class weights
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        w = {0: 1.0, 1: n_neg / max(n_pos, 1)}

        clf.fit(
            X_tr_s, y_tr,
            eval_set=[(X_te_s, y_te)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=100,
            patience=20,
            batch_size=1024,
            virtual_batch_size=256,
            weights=w,
            drop_last=False,
        )
        probs = clf.predict_proba(X_te_s)[:, 1]
        af1 = adjusted_f1(probs, y_te) * 100
        scores.append(af1)
        print(f"  seed {seed}: {af1:.3f}")

    m, s = np.mean(scores), np.std(scores)
    print(f"  → {DISPLAY[task]}: {m:.2f} ± {s:.2f}")
    results.append({"task": task, "display": DISPLAY[task], "method": "TabNet-official",
                    "mean": m, "std": s, "n_seeds": n_seeds,
                    "scores": str(scores)})

df = pd.DataFrame(results)
out = f"{ROOT}/results/tabnet_official_tied.csv"
df.to_csv(out, index=False)
print(f"\n[DONE] saved to {out}")
print(df[["display", "mean", "std"]].to_string())
