#!/usr/bin/env python3
"""Re-run all 15 tasks with official pytorch-tabnet library (correct implementation)."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)

TASKS = [
    "task_pharos_tclin_vs_others",
    "task_pharos_tclin_tchem_vs_others",
    "task_triage_tier1_vs_others",
    "task_triage_tier12_vs_others",
    "task_cancer_druggability",
    "task_cancer_type_specific_target_prioritization",
    "task_pan_cancer_target_prioritization",
    "task_T1_targets_only",
    "task_T1_T2_targets",
    "task_T1_T2_T3_targets",
    "task_sm_bucket1_vs_others",
    "task_sm_bucket123_vs_others",
    "task_ab_bucket1_vs_others",
    "task_ab_bucket123_vs_others",
    "task_protac_bucket1234_vs_others",
]
DISPLAY = {
    "task_pharos_tclin_vs_others": "Clinical Targets",
    "task_pharos_tclin_tchem_vs_others": "Clinical & Chemical",
    "task_triage_tier1_vs_others": "Top-Tier Targets",
    "task_triage_tier12_vs_others": "High-Confidence Targets",
    "task_cancer_druggability": "Cancer-Relevant",
    "task_cancer_type_specific_target_prioritization": "Cancer Type-Specific",
    "task_pan_cancer_target_prioritization": "Pan-Cancer",
    "task_T1_targets_only": "T1 Cancer",
    "task_T1_T2_targets": "T1-T2 Cancer",
    "task_T1_T2_T3_targets": "T1-T3 Cancer",
    "task_sm_bucket1_vs_others": "Small-Molecule (Approved)",
    "task_sm_bucket123_vs_others": "Small-Molecule (Clinical+)",
    "task_ab_bucket1_vs_others": "Antibody (Approved)",
    "task_ab_bucket123_vs_others": "Antibody (Clinical+)",
    "task_protac_bucket1234_vs_others": "PROTAC Targets",
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

results = []
for task in TASKS:
    if task not in labels.columns:
        print(f"[SKIP] {task} not in labels")
        continue
    y_all = labels[task].reindex(genes).values.astype(float)
    valid = ~np.isnan(y_all)
    X = X_all[valid]; y = y_all[valid].astype(int)
    n_seeds = 20 if task == "task_ab_bucket1_vs_others" else 5
    print(f"\n[{DISPLAY[task]}] |P|={y.sum()}, n={valid.sum()}, seeds={n_seeds}")
    scores = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr).astype(np.float32)
        X_te_s = sc.transform(X_te).astype(np.float32)
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        w = {0: 1.0, 1: n_neg / max(n_pos, 1)}
        clf = TabNetClassifier(
            n_d=32, n_a=32, n_steps=5,
            gamma=1.5, lambda_sparse=1e-4,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=None,
            verbose=0, seed=seed,
        )
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
        print(f"  seed {seed}: {af1:.3f}", flush=True)

    m, s = np.mean(scores), np.std(scores)
    print(f"  → {m:.2f} ± {s:.2f}", flush=True)
    results.append({"task": task, "display": DISPLAY[task], "method": "TabNet",
                    "mean": m, "std": s, "n_seeds": n_seeds})
    pd.DataFrame(results).to_csv("results/tabnet_official_all.csv", index=False)

print("\n[DONE]")
df = pd.DataFrame(results)
print(df[["display", "mean", "std"]].to_string())
