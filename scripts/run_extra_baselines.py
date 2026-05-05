#!/usr/bin/env python3
"""
Run additional strong baselines on the 15-task benchmark:
  - LightGBM  (all features, d=750)
  - ResNet-MLP (all features, d=750)
  - FT-Transformer (all features, d=750)
  - XGBoost(bio-only, d=482) - fair comparison with classical ML

All methods use: same 80/20 stratified split, 5 seeds, Adjusted F1.
No PU learning (supervised BCE with class weighting).
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from targetsage import (
    TASKS, TASK_DISPLAY,
    load_bio_features, load_llm_scores, load_llm_embeddings,
    load_labels, build_feature_matrix, get_feature_arrays,
)
from targetsage.metrics import adjusted_f1


# ─── Models ───────────────────────────────────────────────────────────────────

class ResNetBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class ResNetMLP(nn.Module):
    def __init__(self, d_in, d_hidden=256, n_blocks=4, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResNetBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x))).squeeze(-1)


class FeatureTokenizer(nn.Module):
    """Tokenize each feature into a d_emb-dim embedding."""
    def __init__(self, n_features, d_emb):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_emb))
        self.bias   = nn.Parameter(torch.empty(n_features, d_emb))
        nn.init.normal_(self.weight, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, n_features) → (B, n_features, d_emb)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_emb=64, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_emb)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_emb))
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_heads, dim_feedforward=d_emb*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, 1))

    def forward(self, x):
        tokens = self.tokenizer(x)                        # (B, d, emb)
        cls    = self.cls_token.expand(x.size(0), -1, -1) # (B, 1, emb)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, d+1, emb)
        out    = self.transformer(tokens)                  # (B, d+1, emb)
        return self.head(out[:, 0]).squeeze(-1)             # (B,)


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_nn(model, X_tr, y_tr, n_epochs, lr, batch_size, pos_weight, device, verbose=False):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    X = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X))
        total_loss = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X[idx])
            loss = crit(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()
        if verbose and (epoch+1) % 20 == 0:
            print(f"    epoch {epoch+1}/{n_epochs} loss={total_loss:.4f}")


@torch.no_grad()
def predict_nn(model, X, device, batch=4096):
    model.eval()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = []
    for i in range(0, len(X_t), batch):
        logits = model(X_t[i:i+batch])
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run_lgbm(X_tr, y_tr, X_te, seed):
    import lightgbm as lgb
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=63, min_child_samples=5,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=seed, n_jobs=4, verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def run_xgb_bio(X_tr, y_tr, X_te, seed):
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric='logloss', random_state=seed, n_jobs=4, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio",        default="data/gene_features.tsv")
    ap.add_argument("--scores",     default="data/features_llm_structured_scores.csv")
    ap.add_argument("--embeddings", default="data/features_llm_embedding.csv")
    ap.add_argument("--labels",     default="data/gene_labels.tsv")
    ap.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir",     default="results/extra_baselines")
    ap.add_argument("--emb_pca_dim", type=int, default=256)
    ap.add_argument("--methods",    default="lgbm,xgb_bio,resnet,fttransformer",
                    help="Comma-separated list of methods to run")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",")]

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[DATA] Loading features...")
    bio_df   = load_bio_features(args.bio)
    score_df = load_llm_scores(args.scores)
    emb_df   = load_llm_embeddings(args.embeddings)
    label_df = load_labels(args.labels)

    merged = build_feature_matrix(bio_df, score_df, emb_df, label_df)
    Xb_all, Xa_all, Xe_all, _ = get_feature_arrays(merged)

    # Preprocess once globally
    Xb = SimpleImputer(strategy="median").fit_transform(Xb_all)
    Xa = SimpleImputer(strategy="median").fit_transform(Xa_all)
    Xe = SimpleImputer(strategy="median").fit_transform(Xe_all)

    if args.emb_pca_dim > 0 and Xe.shape[1] > args.emb_pca_dim:
        Xe = PCA(n_components=args.emb_pca_dim, random_state=42).fit_transform(Xe)

    Xb = StandardScaler().fit_transform(Xb)
    Xa = StandardScaler().fit_transform(Xa)
    Xe = StandardScaler().fit_transform(Xe)

    X_all = np.hstack([Xb, Xa, Xe])   # all features (bio+attr+emb), d=750
    print(f"  X_bio: {Xb.shape[1]}, X_all: {X_all.shape[1]}")

    all_results = []

    for task in TASKS:
        if task not in merged.columns:
            continue
        y_all = merged[task].to_numpy()
        valid = ~np.isnan(y_all)
        if valid.sum() < 50:
            continue

        y   = y_all[valid].astype(int)
        Xf  = X_all[valid]
        Xb_ = Xb[valid]
        n_pos = int(y.sum())
        if n_pos < 5:
            continue

        display = TASK_DISPLAY.get(task, task)
        print(f"\n[{display}] |P|={n_pos} |U|={len(y)-n_pos}")

        for method in methods:
            seed_scores = []
            for seed in args.seeds:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    Xf if method != "xgb_bio" else Xb_,
                    y, test_size=0.2, stratify=y, random_state=seed
                )
                n_pos_tr = int(y_tr.sum())
                n_neg_tr = len(y_tr) - n_pos_tr
                pos_w    = n_neg_tr / max(n_pos_tr, 1)

                try:
                    if method == "lgbm":
                        probs = run_lgbm(X_tr, y_tr, X_te, seed)

                    elif method == "xgb_bio":
                        probs = run_xgb_bio(X_tr, y_tr, X_te, seed)

                    elif method == "resnet":
                        torch.manual_seed(seed)
                        model = ResNetMLP(X_tr.shape[1], d_hidden=256, n_blocks=4).to(device)
                        train_nn(model, X_tr, y_tr, n_epochs=100, lr=1e-3,
                                 batch_size=512, pos_weight=pos_w, device=device)
                        probs = predict_nn(model, X_te, device)

                    elif method == "fttransformer":
                        torch.manual_seed(seed)
                        # Use PCA to limit feature count for FT-Transformer (speed)
                        n_feat = min(X_tr.shape[1], 128)
                        if X_tr.shape[1] > n_feat:
                            pca = PCA(n_components=n_feat, random_state=seed)
                            X_tr_f = pca.fit_transform(X_tr)
                            X_te_f = pca.transform(X_te)
                        else:
                            X_tr_f, X_te_f = X_tr, X_te
                        model = FTTransformer(
                            n_features=n_feat, d_emb=64, n_heads=8, n_layers=3
                        ).to(device)
                        train_nn(model, X_tr_f, y_tr, n_epochs=100, lr=5e-4,
                                 batch_size=512, pos_weight=pos_w, device=device)
                        probs = predict_nn(model, X_te_f, device)

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    score = adjusted_f1(y_te, probs)
                    seed_scores.append(score)
                    print(f"  {method} seed={seed}: adj_f1={score:.4f}")

                except Exception as e:
                    print(f"  {method} seed={seed}: ERROR {e}")
                    import traceback; traceback.print_exc()

            if seed_scores:
                m = float(np.mean(seed_scores))
                s = float(np.std(seed_scores))
                all_results.append(dict(
                    task=task, display=display, method=method,
                    adj_f1_mean=m, adj_f1_std=s, n_seeds=len(seed_scores),
                ))
                print(f"  {method}: {m:.2f} ± {s:.2f}")

    df = pd.DataFrame(all_results)
    out = os.path.join(args.outdir, "extra_baselines_results.csv")
    df.to_csv(out, index=False)
    print(f"\n[DONE] Saved to {out}")

    # Summary
    pivot = df.pivot_table(index="display", columns="method", values="adj_f1_mean")
    print("\nSummary (Adj F1 %):")
    print(pivot.round(2).to_string())
    macros = df.groupby("method")["adj_f1_mean"].mean()
    print("\nMacro averages:")
    print(macros.round(2).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
