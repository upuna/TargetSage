#!/usr/bin/env python3
"""
Train each baseline on ALL 2025 positives (no held-out set), then rank all genes.
Outputs one ranking CSV per method, same format as inference.py:
    rank, Gene_Symbol, score

Usage:
    python scripts/run_baseline_inference.py \
        --task task_pharos_tclin_vs_others \
        --outdir results/baseline_inference_2025
"""

import os, sys, argparse
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from targetsage import load_bio_features, load_labels, TASKS

# deep tabular
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_classical():
    return {
        "LR":  LogisticRegression(max_iter=5000, C=1.0),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=1),
        "NB":  GaussianNB(),
        "RF":  RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=2),
        "GB":  GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
    }


class ResNetBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.bn   = nn.BatchNorm1d(d)
    def forward(self, x):
        return x + self.drop(F.relu(self.fc2(F.relu(self.fc1(self.bn(x))))))

class SimpleResNet(nn.Module):
    def __init__(self, in_dim, d=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d)
        self.blocks = nn.Sequential(*[ResNetBlock(d, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        return self.head(self.blocks(F.relu(self.embed(x)))).squeeze(-1)


def train_resnet(X, y, device, epochs=50, lr=1e-3, batch=256):
    model = SimpleResNet(X.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32).to(device)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt))
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(Xt.to(device))).cpu().numpy()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio",    default="data/gene_features.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--task",   default="task_pharos_tclin_vs_others")
    ap.add_argument("--outdir", default="results/baseline_inference_2025")
    ap.add_argument("--resnet", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[DATA] Loading ...")
    bio_df   = load_bio_features(args.bio)
    label_df = load_labels(args.labels)
    merged   = label_df.merge(bio_df, on="Gene_Symbol", how="inner")
    DRUG_FEAT_EXCL = {"monoclonalCount", "antibodyCount", "DGIdb_interaction_types",
                      "ctd_uniqueInteractions", "ctd_otherInteractionsCount"}
    bio_cols = [c for c in merged.columns
                if c not in ["Gene_Symbol"] + TASKS and c not in DRUG_FEAT_EXCL]
    print(f"  Using {len(bio_cols)} features (excluded {len(DRUG_FEAT_EXCL)} drug-related)")

    X_raw = merged[bio_cols].to_numpy(dtype=float)
    imp   = SimpleImputer(strategy="median").fit(X_raw)
    scl   = StandardScaler().fit(imp.transform(X_raw))
    X     = scl.transform(imp.transform(X_raw))
    genes = merged["Gene_Symbol"].tolist()

    if args.task not in merged.columns:
        print(f"[ERROR] task {args.task} not found"); return

    y = merged[args.task].to_numpy(dtype=float)
    valid = ~np.isnan(y)
    X_fit, y_fit = X[valid], y[valid].astype(int)
    print(f"  {len(genes)} genes total, {y_fit.sum()} positives for {args.task}")

    def save_ranking(name, scores_all):
        df = pd.DataFrame({"Gene_Symbol": genes, "score": scores_all})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
        path = os.path.join(args.outdir, f"{name}_ranking.csv")
        df.to_csv(path, index=False)
        print(f"  saved {path}")

    # ── Classical baselines ───────────────────────────────────────────────────
    for name, clf in make_classical().items():
        out_path = os.path.join(args.outdir, f"{name}_ranking.csv")
        if os.path.exists(out_path):
            print(f"[{name}] already done, skipping")
            continue
        print(f"[{name}] fitting ...")
        # Tree-based models use cross-validated isotonic calibration (more robust)
        # Linear/other models use prefit sigmoid calibration
        if name in ("RF", "GB"):
            cal = CalibratedClassifierCV(clf, cv=5, method="isotonic")
        else:
            clf.fit(X_fit, y_fit)
            cal = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
        cal.fit(X_fit, y_fit)
        scores_all = cal.predict_proba(X)[:, 1]
        save_ranking(name, scores_all)

    # ── ResNet ────────────────────────────────────────────────────────────────
    if args.resnet:
        out_path = os.path.join(args.outdir, "ResNet_ranking.csv")
        if os.path.exists(out_path):
            print("[ResNet] already done, skipping")
        else:
            print("[ResNet] training ...")
            scores_all = train_resnet(X, np.where(valid, y, 0).astype(float), device)
            save_ranking("ResNet", scores_all)

    print("[DONE]")


if __name__ == "__main__":
    main()
