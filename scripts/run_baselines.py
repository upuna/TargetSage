#!/usr/bin/env python3
"""
TargetSage — Baseline Reproduction
=====================================
Reproduces all ten baseline results from Table 2 of the paper:
  - 7 classical ML methods: LR, SVM, KNN, NB, RF, GB, MLP
  - 3 deep tabular models: ResNet, TabNet, FT-Transformer

All methods use bio-only features (482 dims after imputation and scaling),
the same 80/20 stratified train/test splits as train.py, and Adjusted F1
as the evaluation metric.

Main results from the paper (macro-avg Adjusted F1):
  Best deep tabular baseline (ResNet):  8.1%
  Best classical ML baseline (GB):      7.6%
  TargetSage (train.py):               11.1%

Usage
-----
    # Classical ML only (fast, CPU)
    python scripts/run_baselines.py --mode classical

    # Deep tabular only (requires GPU or ~20 min on CPU)
    python scripts/run_baselines.py --mode deep

    # Both (default)
    python scripts/run_baselines.py

    # Quick test with fewer seeds
    python scripts/run_baselines.py --seeds 0 1 2

Output
------
    results/baselines/classical_baselines_results.csv
    results/baselines/deep_baselines_results.csv
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from targetsage import TASKS, TASK_DISPLAY, load_bio_features, load_labels
from targetsage.metrics import adjusted_f1


# ─────────────────────────────────────────────────────────────────────────────
# Classical ML classifiers (Section 4.1 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def make_classical_classifiers():
    return {
        "LR":  LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=4),
        "NB":  GaussianNB(),
        "RF":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=4),
        "GB":  GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deep tabular models (Section 4.1 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

class ResNetBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class ResNet(nn.Module):
    """ResNet-style MLP with skip connections (He et al., 2016)."""
    def __init__(self, d_in, d_hidden=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.stem   = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResNetBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Linear(d_hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


class FeatureTokenizer(nn.Module):
    """Project each scalar feature to a d_emb-dim token (FT-Transformer tokenizer)."""
    def __init__(self, n_features, d_emb):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_emb))
        self.bias   = nn.Parameter(torch.empty(n_features, d_emb))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021)."""
    def __init__(self, n_features, d_emb=64, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_emb)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_emb))
        enc_layer      = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, 1))

    def forward(self, x):
        tokens = self.tokenizer(x)
        cls    = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out    = self.transformer(tokens)
        return self.head(out[:, 0]).squeeze(-1)


class TabNetSimple(nn.Module):
    """
    Simplified TabNet using sequential attention over features
    (Arik & Pfister, 2021).
    """
    def __init__(self, d_in, d_hidden=128, n_steps=3, dropout=0.1):
        super().__init__()
        self.n_steps = n_steps
        self.shared  = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.BatchNorm1d(d_hidden),
        )
        self.step_attn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_hidden, d_in), nn.Softmax(dim=-1))
            for _ in range(n_steps)
        ])
        self.step_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(n_steps)
        ])
        self.head = nn.Linear(d_hidden, 1)
        self.bn   = nn.BatchNorm1d(d_in)

    def forward(self, x):
        x_bn = self.bn(x)
        h    = self.shared(x_bn)
        agg  = torch.zeros(x.size(0), h.size(1), device=x.device)
        for attn, fc in zip(self.step_attn, self.step_fc):
            mask = attn(h)
            agg  = agg + fc(mask * x_bn)
        return self.head(agg).squeeze(-1)


def train_nn(model, X_tr, y_tr, n_epochs, lr, batch_size, pos_weight, device):
    model.train()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    X = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    for _ in range(n_epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx    = perm[i:i + batch_size]
            logits = model(X[idx])
            loss   = crit(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


@torch.no_grad()
def predict_nn(model, X, device, batch=4096):
    model.eval()
    X_t   = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = []
    for i in range(0, len(X_t), batch):
        logits = model(X_t[i:i + batch])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_classical(X, y_by_task, seeds, outdir):
    classifiers = make_classical_classifiers()
    all_results = []

    for task in TASKS:
        if task not in y_by_task:
            continue
        y_all   = y_by_task[task]
        valid   = ~np.isnan(y_all)
        y       = y_all[valid].astype(int)
        Xv      = X[valid]
        n_pos   = int(y.sum())
        display = TASK_DISPLAY.get(task, task)
        if n_pos < 5:
            continue
        print(f"\n[Classical] {display}  |P|={n_pos}")

        for clf_name, clf_tmpl in classifiers.items():
            seed_scores = []
            for seed in seeds:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xv, y, test_size=0.2, stratify=y, random_state=seed
                )
                try:
                    from sklearn.base import clone
                    clf = clone(clf_tmpl)
                    clf.fit(X_tr, y_tr)
                    probs = clf.predict_proba(X_te)[:, 1]
                    seed_scores.append(adjusted_f1(y_te, probs))
                except Exception as e:
                    print(f"  {clf_name} seed={seed}: ERROR {e}")
            if seed_scores:
                all_results.append({
                    "task": task, "display": display, "method": clf_name,
                    "adj_f1_mean": float(np.mean(seed_scores)),
                    "adj_f1_std":  float(np.std(seed_scores)),
                })
                print(f"  {clf_name}: {np.mean(seed_scores)*100:.2f} ± {np.std(seed_scores)*100:.2f}%")

    out = os.path.join(outdir, "classical_baselines_results.csv")
    pd.DataFrame(all_results).to_csv(out, index=False)
    print(f"\n[Classical] Saved → {out}")


def run_deep(X, y_by_task, seeds, outdir, n_epochs=50, batch=512, lr=1e-3):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    d_in      = X.shape[1]
    deep_specs = {
        "ResNet":         lambda: ResNet(d_in),
        "TabNet":         lambda: TabNetSimple(d_in),
        "FT-Transformer": lambda: FTTransformer(d_in),
    }
    all_results = []

    for task in TASKS:
        if task not in y_by_task:
            continue
        y_all   = y_by_task[task]
        valid   = ~np.isnan(y_all)
        y       = y_all[valid].astype(int)
        Xv      = X[valid]
        n_pos   = int(y.sum())
        display = TASK_DISPLAY.get(task, task)
        if n_pos < 5:
            continue
        print(f"\n[Deep] {display}  |P|={n_pos}")

        for model_name, model_fn in deep_specs.items():
            seed_scores = []
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xv, y, test_size=0.2, stratify=y, random_state=seed
                )
                pos_weight = float((y_tr == 0).sum() / max(y_tr.sum(), 1))
                try:
                    model = model_fn().to(device)
                    train_nn(model, X_tr, y_tr, n_epochs, lr, batch, pos_weight, device)
                    probs = predict_nn(model, X_te, device)
                    seed_scores.append(adjusted_f1(y_te, probs))
                except Exception as e:
                    print(f"  {model_name} seed={seed}: ERROR {e}")
            if seed_scores:
                all_results.append({
                    "task": task, "display": display, "method": model_name,
                    "adj_f1_mean": float(np.mean(seed_scores)),
                    "adj_f1_std":  float(np.std(seed_scores)),
                })
                print(f"  {model_name}: {np.mean(seed_scores)*100:.2f} ± {np.std(seed_scores)*100:.2f}%")

    out = os.path.join(outdir, "deep_baselines_results.csv")
    pd.DataFrame(all_results).to_csv(out, index=False)
    print(f"\n[Deep] Saved → {out}")


def main():
    ap = argparse.ArgumentParser(
        description="TargetSage baseline reproduction (classical + deep tabular)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--bio",    default="data/gene_features.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--seeds",  type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", default="results/baselines")
    ap.add_argument("--mode",   choices=["classical", "deep", "all"], default="all",
                    help="Which baselines to run")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[DATA] Loading...")
    bio_df   = load_bio_features(args.bio)
    label_df = load_labels(args.labels)
    merged   = label_df.merge(bio_df, on="Gene_Symbol", how="inner")
    bio_cols = [c for c in merged.columns if c not in ["Gene_Symbol"] + TASKS]

    Xb_raw = merged[bio_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(Xb_raw)
    )
    print(f"  {len(merged)} genes × {X.shape[1]} bio features")

    y_by_task = {t: merged[t].to_numpy(dtype=float) for t in TASKS if t in merged.columns}

    if args.mode in ("classical", "all"):
        run_classical(X, y_by_task, args.seeds, args.outdir)
    if args.mode in ("deep", "all"):
        run_deep(X, y_by_task, args.seeds, args.outdir)


if __name__ == "__main__":
    main()
