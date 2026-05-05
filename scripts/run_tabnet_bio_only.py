#!/usr/bin/env python3
"""TabNet on bio-only features (d=482), 15 tasks, 5 seeds (20 for Ab Approved)."""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)

TASKS = [
    "task_pharos_tclin_vs_others", "task_pharos_tclin_tchem_vs_others",
    "task_triage_tier1_vs_others", "task_triage_tier12_vs_others",
    "task_cancer_druggability", "task_cancer_type_specific_target_prioritization",
    "task_pan_cancer_target_prioritization", "task_T1_targets_only",
    "task_T1_T2_targets", "task_T1_T2_T3_targets",
    "task_sm_bucket1_vs_others", "task_sm_bucket123_vs_others",
    "task_ab_bucket1_vs_others", "task_ab_bucket123_vs_others",
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

class TabNetSimple(nn.Module):
    def __init__(self, d_in, d_h=128, n_steps=3, dropout=0.1):
        super().__init__()
        self.attn = nn.ModuleList([nn.Sequential(
            nn.Linear(d_in, d_in), nn.BatchNorm1d(d_in), nn.GELU(),
            nn.Linear(d_in, d_in), nn.Sigmoid(),
        ) for _ in range(n_steps)])
        self.feat = nn.ModuleList([nn.Sequential(
            nn.Linear(d_in, d_h), nn.BatchNorm1d(d_h), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_h, d_h), nn.GELU(),
        ) for _ in range(n_steps)])
        self.head = nn.Linear(d_h, 1)
    def forward(self, x):
        out = 0
        for a, f in zip(self.attn, self.feat):
            out = out + f(x * a(x))
        return self.head(out).squeeze(-1)

print("[DATA] loading...")
bio = pd.read_csv("data/gene_features.tsv", sep="\t").set_index("Gene_Symbol")
bio = bio.apply(pd.to_numeric, errors='coerce').fillna(0.0)
labels = pd.read_csv("data/gene_labels.tsv", sep="\t").set_index("Gene_Symbol")
genes = sorted(set(bio.index) & set(labels.index))
X_all = np.nan_to_num(bio.reindex(genes).values.astype(float), nan=0.0)
print(f"  {len(genes)} genes, X shape: {X_all.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_one(X_tr, y_tr, X_te, y_te, epochs=80, lr=1e-3, bs=512):
    pos_w = torch.tensor([(len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)], dtype=torch.float32, device=device)
    model = TabNetSimple(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xe = torch.from_numpy(X_te.astype(np.float32)).to(device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = idx[i:i+bs]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xe)).cpu().numpy()
    return adjusted_f1(prob, y_te) * 100

results = []
for task in TASKS:
    if task not in labels.columns: continue
    y_all = labels[task].reindex(genes).values.astype(float)
    valid = ~np.isnan(y_all)
    X = X_all[valid]; y = y_all[valid].astype(int)
    n_seeds = 20 if task == "task_ab_bucket1_vs_others" else 5
    print(f"\n[{DISPLAY[task]}] |P|={y.sum()}, n_seeds={n_seeds}")
    scores = []
    for seed in range(n_seeds):
        np.random.seed(seed); torch.manual_seed(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sc = StandardScaler().fit(X_tr); X_tr = sc.transform(X_tr); X_te = sc.transform(X_te)
        af1 = train_one(X_tr, y_tr, X_te, y_te)
        scores.append(af1)
    m, s = np.mean(scores), np.std(scores)
    print(f"  TabNet: {m:.2f} ± {s:.2f}")
    results.append({"task": task, "display": DISPLAY[task], "method": "TabNet",
                    "mean": m, "std": s, "n_seeds": n_seeds})
    pd.DataFrame(results).to_csv("results/tabnet_bio_only.csv", index=False)

print("\nDONE")
