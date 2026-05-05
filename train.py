#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetSage — Module M3: TargetSage Training and Evaluation
==========================================================
This script is the main entry point for reproducing the paper's core results.
It trains TargetSage (the PU-aware scoring model) across all 15 benchmark
tasks using stratified 80/20 splits, nnPU learning with a hybrid class prior,
and semantic reweighting, then reports Adjusted F1 for each task.

Pipeline per (task, seed)
--------------------------
1. Load and align all modality features (bio, attr, emb) on Gene_Symbol.
2. Stratified 80/20 train/test split.
3. Preprocessing: median imputation → optional PCA on embeddings → StandardScale.
4. Stage 1 — Warmup (10 epochs, BCE with class-imbalance pos_weight).
   Goal: give the model a reasonable initialization before nnPU training.
5. Prior estimation: Elkan-Noto method on the warmup model's outputs.
   Hybrid prior: π = α·π_data + (1-α)·π_llm  (α=0.6 by default).
6. Stage 2 — nnPU training (25 epochs).
   Semantic weights: w_j = β·D(z_j) + (1-β)·cos(h_e_j, centroid_pos)
   (β=0.6 by default).
7. Evaluate on the held-out 20% using Adjusted F1 = R_soft²/p̄.

Usage
-----
    # All 15 tasks, 5 seeds (~3h on RTX 3090)
    python train.py

    # Quick test (2 tasks, 1 seed)
    python train.py --tasks task_pharos_tclin_vs_others,task_T1_targets_only --seeds 0

    # Change LLM backend (affects which precomputed features to load)
    python train.py --llm_backend azure

Output
------
    results/train_<timestamp>/
        results.csv   — per-task mean ± std Adjusted F1 over seeds
        raw.csv       — per-task per-seed rows
        config.json   — full run configuration

Expected Results (macro-average over 15 tasks, 5 seeds):
    TargetSage:  Adjusted F1 = 11.1%  (vs. best baseline RF = 8.1%)
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from targetsage import (
    TargetSage, nnpu_loss, adjusted_f1,
    TASKS, TASK_DISPLAY,
    load_bio_features, load_llm_scores, load_llm_embeddings,
    load_labels, load_prior_map,
    build_feature_matrix, get_feature_arrays,
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, Xb, Xa, Xe, device, batch=4096):
    model.eval()
    probs = []
    for i in range(0, len(Xb), batch):
        lp, _, _ = model.forward_logits(
            torch.from_numpy(Xb[i:i+batch]).to(device),
            torch.from_numpy(Xa[i:i+batch]).to(device),
            torch.from_numpy(Xe[i:i+batch]).to(device),
        )
        probs.append(torch.sigmoid(lp).cpu().numpy())
    return np.concatenate(probs)


def warmup(model, Xb, Xa, Xe, y, epochs, lr, batch_size, device):
    """Stage 1: weighted BCE on observed labels (natural proportions)."""
    if epochs <= 0:
        return
    y = y.astype(int)
    n_pos, n_unl = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 10 or n_unl < 10:
        return

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_unl / max(n_pos, 1)], device=device)
    )
    perm = np.random.permutation(len(y))
    for _ in range(epochs):
        for i in range(0, len(y), batch_size):
            idx = perm[i:i+batch_size]
            lp, _, _ = model.forward_logits(
                torch.from_numpy(Xb[idx]).to(device),
                torch.from_numpy(Xa[idx]).to(device),
                torch.from_numpy(Xe[idx]).to(device),
            )
            opt.zero_grad()
            bce(lp, torch.from_numpy(y[idx].astype(np.float32)).to(device)).backward()
            opt.step()


def train_nnpu(model, Xb, Xa, Xe, y, pi, epochs, lr, batch_size, beta, device):
    """
    Stage 2: nnPU with semantic-guided unlabeled reweighting.
    w_j = β·sigmoid(D(z_j)) + (1-β)·cos(h_e_j, centroid_pos)
    """
    if epochs <= 0:
        return

    idx_p = np.where(y == 1)[0]
    idx_u = np.where(y == 0)[0]
    if len(idx_p) < 10 or len(idx_u) < 10:
        return

    # Pre-compute semantic weights (once, before training loop)
    model.eval()
    with torch.no_grad():
        lp, _, h_e = model.forward_logits(
            torch.from_numpy(Xb).to(device),
            torch.from_numpy(Xa).to(device),
            torch.from_numpy(Xe).to(device),
        )
        prob = torch.sigmoid(lp)
        centroid = F.normalize(h_e[idx_p].mean(0, keepdim=True), dim=1)
        h_u_norm = F.normalize(h_e[idx_u], dim=1)
        sim_u    = ((h_u_norm * centroid).sum(1) + 1.0) / 2.0  # [0,1]
        w_u      = torch.clamp(beta * prob[idx_u] + (1 - beta) * sim_u, 0, 1).cpu().numpy()

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    u_map = {int(idx_u[i]): i for i in range(len(idx_u))}

    n_batch = max(1, min(len(idx_p), len(idx_u)) // batch_size)
    for _ in range(epochs):
        for _ in range(n_batch):
            bp = np.random.choice(idx_p, batch_size, replace=True)
            bu = np.random.choice(idx_u, batch_size, replace=True)

            lp_p, _, _ = model.forward_logits(
                torch.from_numpy(Xb[bp]).to(device),
                torch.from_numpy(Xa[bp]).to(device),
                torch.from_numpy(Xe[bp]).to(device),
            )
            lp_u, _, _ = model.forward_logits(
                torch.from_numpy(Xb[bu]).to(device),
                torch.from_numpy(Xa[bu]).to(device),
                torch.from_numpy(Xe[bu]).to(device),
            )
            w_batch = torch.from_numpy(
                w_u[[u_map[int(g)] for g in bu]]
            ).to(device).float()

            loss = nnpu_loss(lp_p, lp_u, pi=pi, w_u=w_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()


def estimate_prior_en(prob_tr, y_tr, seed, calib_frac, pi_cap):
    """Elkan-Noto prior estimate: pi_data = mean_U(D) / mean_Pcal(D)."""
    idx_p = np.where(y_tr == 1)[0]
    idx_u = np.where(y_tr == 0)[0]
    if len(idx_p) < 5 or len(idx_u) < 5:
        return float(np.mean(y_tr == 1))
    rng      = np.random.RandomState(seed)
    cal_idx  = rng.permutation(idx_p)[:max(1, int(calib_frac * len(idx_p)))]
    c        = float(np.clip(np.mean(prob_tr[cal_idx]), 1e-3, 1.0))
    pi_raw   = float(np.mean(prob_tr[idx_u]))
    return float(np.clip(pi_raw / c, float(np.mean(y_tr == 1)), pi_cap))


def preprocess_split(Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te,
                     emb_pca_dim, seed, attr_pca_dim=0):
    """Impute → optional PCA on attrs/embeddings → StandardScale."""
    imp_b = SimpleImputer(strategy="median")
    imp_a = SimpleImputer(strategy="median")
    imp_e = SimpleImputer(strategy="median")

    Xb_tr = imp_b.fit_transform(Xb_tr); Xb_te = imp_b.transform(Xb_te)
    Xa_tr = imp_a.fit_transform(Xa_tr); Xa_te = imp_a.transform(Xa_te)
    Xe_tr = imp_e.fit_transform(Xe_tr); Xe_te = imp_e.transform(Xe_te)

    if attr_pca_dim > 0 and Xa_tr.shape[1] > attr_pca_dim:
        k = min(attr_pca_dim, Xa_tr.shape[1])
        pca_a = PCA(n_components=k, random_state=seed)
        Xa_tr = pca_a.fit_transform(Xa_tr)
        Xa_te = pca_a.transform(Xa_te)

    if emb_pca_dim > 0:
        k = min(emb_pca_dim, Xe_tr.shape[1])
        pca = PCA(n_components=k, random_state=seed)
        Xe_tr = pca.fit_transform(Xe_tr)
        Xe_te = pca.transform(Xe_te)

    sc_b = StandardScaler(); sc_a = StandardScaler(); sc_e = StandardScaler()
    Xb_tr = sc_b.fit_transform(Xb_tr); Xb_te = sc_b.transform(Xb_te)
    Xa_tr = sc_a.fit_transform(Xa_tr); Xa_te = sc_a.transform(Xa_te)
    Xe_tr = sc_e.fit_transform(Xe_tr); Xe_te = sc_e.transform(Xe_te)

    return Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TargetSage train mode — 15-task evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--bio",        default="data/gene_features.tsv")
    ap.add_argument("--scores",     default="data/features_llm_structured_scores.csv")
    ap.add_argument("--embeddings", default="data/features_llm_embedding.csv")
    ap.add_argument("--labels",     default="data/gene_labels.tsv")
    ap.add_argument("--prior_csv",  default="data/prior_task_summary.csv")
    ap.add_argument("--outdir",     default="results")

    # Experiment
    ap.add_argument("--tasks",  default=",".join(TASKS),
                    help="Comma-separated task keys (default: all 15)")
    ap.add_argument("--seeds",  default="0,1,2,3,4")

    # Model
    ap.add_argument("--d_latent",    type=int,   default=256)
    ap.add_argument("--head_h",      type=int,   default=512)
    ap.add_argument("--dropout",     type=float, default=0.2)
    ap.add_argument("--emb_pca_dim",  type=int,   default=256)
    ap.add_argument("--attr_pca_dim", type=int,   default=0,
                    help="PCA dim for LLM attrs (0=off). Use >0 for agent attrs with many cols.")
    ap.add_argument("--fusion",      default="gated", choices=["gated","concat","sum"])

    # Training
    ap.add_argument("--warmup_epochs", type=int,   default=10)
    ap.add_argument("--nnpu_epochs",   type=int,   default=25)
    ap.add_argument("--batch_size",    type=int,   default=512)
    ap.add_argument("--lr",            type=float, default=2e-4)

    # Prior
    ap.add_argument("--alpha",      type=float, default=0.6,
                    help="Hybrid prior: pi = alpha*pi_data + (1-alpha)*pi_llm")
    ap.add_argument("--pi_cap",     type=float, default=0.10)
    ap.add_argument("--calib_frac", type=float, default=0.30)

    # Semantic reweighting
    ap.add_argument("--beta", type=float, default=0.6,
                    help="w = beta*D(z) + (1-beta)*cos_sim")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    # ---- Setup ----
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"train_{ts}")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    seeds     = [int(s) for s in args.seeds.split(",") if s.strip()]
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # ---- Load data ----
    print("[INFO] Loading data ...")
    bio_df   = load_bio_features(args.bio)
    score_df = load_llm_scores(args.scores)
    emb_df   = load_llm_embeddings(args.embeddings)
    label_df = load_labels(args.labels)
    prior_map = load_prior_map(args.prior_csv)

    merged = build_feature_matrix(bio_df, score_df, emb_df, label_df)
    Xb_all, Xa_all, Xe_all, genes_all = get_feature_arrays(merged)
    print(f"[INFO] Aligned genes: {len(merged)}  "
          f"bio={Xb_all.shape[1]}  attr={Xa_all.shape[1]}  emb={Xe_all.shape[1]}")

    # ---- Run ----
    raw_rows = []

    for task in task_list:
        if task not in merged.columns:
            print(f"[WARN] skipping {task}: column not found"); continue

        y_all = merged[task].to_numpy()
        valid = ~np.isnan(y_all)
        if valid.sum() < 50 or len(np.unique(y_all[valid])) < 2:
            print(f"[WARN] skipping {task}: insufficient labels"); continue

        Xb = Xb_all[valid]; Xa = Xa_all[valid]; Xe = Xe_all[valid]
        y  = y_all[valid].astype(int)

        display    = TASK_DISPLAY.get(task, task)
        pi_llm     = prior_map.get(display, np.nan)

        for seed in seeds:
            # Stratified split
            try:
                bc = np.bincount(y)
                strat = y if (len(bc) >= 2 and bc.min() >= 2) else None
            except Exception:
                strat = None

            Xb_tr, Xb_te, Xa_tr, Xa_te, Xe_tr, Xe_te, y_tr, y_te = train_test_split(
                Xb, Xa, Xe, y, test_size=0.2, random_state=seed, stratify=strat
            )

            Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te = preprocess_split(
                Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te,
                args.emb_pca_dim, seed, attr_pca_dim=args.attr_pca_dim
            )

            # Build model
            model = TargetSage(
                d_bio=Xb_tr.shape[1], d_attr=Xa_tr.shape[1], d_emb=Xe_tr.shape[1],
                d_latent=args.d_latent, head_h=args.head_h, dropout=args.dropout,
                fusion=args.fusion,
            ).to(args.device)

            # Stage 1: warmup
            warmup(model, Xb_tr, Xa_tr, Xe_tr, y_tr,
                   args.warmup_epochs, args.lr, args.batch_size, args.device)

            # Prior estimation (EN method)
            prob_tr  = predict(model, Xb_tr, Xa_tr, Xe_tr, args.device)
            pi_data  = estimate_prior_en(prob_tr, y_tr, seed, args.calib_frac, args.pi_cap)
            pi_used  = (
                float(args.alpha * pi_data + (1 - args.alpha) * pi_llm)
                if not np.isnan(pi_llm) else pi_data
            )
            pi_used  = float(np.clip(pi_used, float(np.mean(y_tr == 1)), args.pi_cap))

            # Stage 2: nnPU
            train_nnpu(model, Xb_tr, Xa_tr, Xe_tr, y_tr,
                       pi=pi_used, epochs=args.nnpu_epochs,
                       lr=args.lr, batch_size=args.batch_size,
                       beta=args.beta, device=args.device)

            # Evaluate
            prob_te = predict(model, Xb_te, Xa_te, Xe_te, args.device)
            score   = adjusted_f1(y_te, prob_te)

            raw_rows.append(dict(task=task, display=display, seed=seed,
                                 adj_f1=score, pi_data=pi_data,
                                 pi_llm=pi_llm, pi_used=pi_used))
            print(f"  [{display}] seed={seed}  adj_f1={score:.4f}  pi_used={pi_used:.4f}")

    # ---- Aggregate and save ----
    raw_df  = pd.DataFrame(raw_rows)
    agg_df  = (raw_df.groupby(["task", "display"])
               .agg(adj_f1_mean=("adj_f1","mean"), adj_f1_std=("adj_f1","std"),
                    n_seeds=("seed","nunique"))
               .reset_index()
               .sort_values("adj_f1_mean", ascending=False))

    raw_df.to_csv(os.path.join(outdir, "raw.csv"),     index=False)
    agg_df.to_csv(os.path.join(outdir, "results.csv"), index=False)

    print("\n" + "=" * 65)
    print(f"{'Task':<35} {'Adj F1 (mean±std)':>20}")
    print("-" * 65)
    for _, row in agg_df.iterrows():
        print(f"  {row['display']:<33} {row['adj_f1_mean']*100:>6.2f} ± {row['adj_f1_std']*100:.2f} %")
    macro = agg_df["adj_f1_mean"].mean() * 100
    print("-" * 65)
    print(f"  {'Macro-Average':<33} {macro:>6.2f} %")
    print("=" * 65)
    print(f"\n[DONE] Results saved to {outdir}/")


if __name__ == "__main__":
    main()
