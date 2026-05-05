#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetSage — Module M3: Genome-Wide Target Ranking (Inference Mode)
===================================================================
Trains TargetSage on ALL labeled positives (no held-out test set), then
scores every gene in the genome and outputs a ranked list ordered by the
predicted druggability probability.

Difference from train.py
--------------------------
train.py: 80/20 split → evaluate on 20% held-out set → Adjusted F1 score.
inference.py: train on 100% of labeled data → score all ~19 032 genes.

This script is intended for generating the ranked gene lists reported in
the paper's case studies (Section 4.3) and for producing submission files
for external validation.

Usage
-----
    # Rank all genes for a single task
    python inference.py --task task_pharos_tclin_vs_others

    # Rank for multiple tasks (one output file per task)
    python inference.py --task task_T1_targets_only,task_cancer_druggability

    # Save model checkpoints for further analysis
    python inference.py --task task_pharos_tclin_vs_others --save_model

    # Save only the top 500 genes
    python inference.py --task task_T1_targets_only --top_k 500

Output
------
    results/inference_<timestamp>/
        <task>_ranking.csv   — all genes ranked by TargetSage score
                               columns: rank, Gene_Symbol, targetsage_score, label
        config.json          — full run configuration
        <task>_model.pt      — (if --save_model) model state dict
        <task>_top<K>.csv    — (if --top_k > 0) top-K genes
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
# Helpers (shared with train.py)
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
    if epochs <= 0:
        return
    idx_p = np.where(y == 1)[0]
    idx_u = np.where(y == 0)[0]
    if len(idx_p) < 10 or len(idx_u) < 10:
        return

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
        sim_u    = ((h_u_norm * centroid).sum(1) + 1.0) / 2.0
        w_u      = torch.clamp(beta * prob[idx_u] + (1 - beta) * sim_u, 0, 1).cpu().numpy()

    model.train()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
            w_batch = torch.from_numpy(w_u[[u_map[int(g)] for g in bu]]).to(device).float()
            loss = nnpu_loss(lp_p, lp_u, pi=pi, w_u=w_batch)
            opt.zero_grad(); loss.backward(); opt.step()


def estimate_prior_en(prob, y, seed, calib_frac, pi_cap):
    idx_p = np.where(y == 1)[0]
    idx_u = np.where(y == 0)[0]
    if len(idx_p) < 5 or len(idx_u) < 5:
        return float(np.mean(y == 1))
    cal = np.random.RandomState(seed).permutation(idx_p)[:max(1, int(calib_frac * len(idx_p)))]
    c   = float(np.clip(np.mean(prob[cal]), 1e-3, 1.0))
    return float(np.clip(np.mean(prob[idx_u]) / c, float(np.mean(y == 1)), pi_cap))


def preprocess_full(Xb, Xa, Xe, emb_pca_dim, seed):
    """Impute → PCA → Scale on the full dataset (inference uses all data)."""
    imp_b = SimpleImputer(strategy="median")
    imp_a = SimpleImputer(strategy="median")
    imp_e = SimpleImputer(strategy="median")
    Xb = imp_b.fit_transform(Xb)
    Xa = imp_a.fit_transform(Xa)
    Xe = imp_e.fit_transform(Xe)

    if emb_pca_dim > 0:
        k  = min(emb_pca_dim, Xe.shape[1])
        Xe = PCA(n_components=k, random_state=seed).fit_transform(Xe)

    Xb = StandardScaler().fit_transform(Xb)
    Xa = StandardScaler().fit_transform(Xa)
    Xe = StandardScaler().fit_transform(Xe)
    return Xb, Xa, Xe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="TargetSage inference mode — full-genome ranking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--bio",        default="data/gene_features.tsv")
    ap.add_argument("--scores",     default="data/features_llm_structured_scores.csv")
    ap.add_argument("--embeddings", default="data/features_llm_embedding.csv")
    ap.add_argument("--labels",     default="data/gene_labels.tsv")
    ap.add_argument("--prior_csv",  default="data/prior_task_summary.csv")
    ap.add_argument("--outdir",     default="results")

    # Task(s)
    ap.add_argument("--task", default="task_pharos_tclin_vs_others",
                    help="Comma-separated task key(s) to rank")

    # Model
    ap.add_argument("--d_latent",    type=int,   default=256)
    ap.add_argument("--head_h",      type=int,   default=512)
    ap.add_argument("--dropout",     type=float, default=0.2)
    ap.add_argument("--emb_pca_dim", type=int,   default=256)
    ap.add_argument("--fusion",      default="gated", choices=["gated","concat","sum"])

    # Training
    ap.add_argument("--warmup_epochs", type=int,   default=10)
    ap.add_argument("--nnpu_epochs",   type=int,   default=30)
    ap.add_argument("--batch_size",    type=int,   default=512)
    ap.add_argument("--lr",            type=float, default=2e-4)
    ap.add_argument("--seed",          type=int,   default=42)

    # Prior
    ap.add_argument("--alpha",      type=float, default=0.6)
    ap.add_argument("--pi_cap",     type=float, default=0.10)
    ap.add_argument("--calib_frac", type=float, default=0.30)
    ap.add_argument("--beta",       type=float, default=0.6)

    # Output
    ap.add_argument("--save_model", action="store_true",
                    help="Save model checkpoint (.pt) for each task")
    ap.add_argument("--top_k",      type=int, default=0,
                    help="Also save a top-K CSV (0 = full ranking only)")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    # ---- Setup ----
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"inference_{ts}")
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    task_list = [t.strip() for t in args.task.split(",") if t.strip()]

    # ---- Load data ----
    print("[INFO] Loading data ...")
    bio_df    = load_bio_features(args.bio)
    score_df  = load_llm_scores(args.scores)
    emb_df    = load_llm_embeddings(args.embeddings)
    label_df  = load_labels(args.labels)
    prior_map = load_prior_map(args.prior_csv)

    merged = build_feature_matrix(bio_df, score_df, emb_df, label_df)
    Xb_all, Xa_all, Xe_all, genes_all = get_feature_arrays(merged)
    print(f"[INFO] Genome: {len(merged)} genes  "
          f"bio={Xb_all.shape[1]}  attr={Xa_all.shape[1]}  emb={Xe_all.shape[1]}")

    # ---- Per-task ranking ----
    for task in task_list:
        if task not in merged.columns:
            print(f"[WARN] {task}: column not found, skipping"); continue

        y_all = merged[task].to_numpy()
        valid = ~np.isnan(y_all)
        if valid.sum() < 50:
            print(f"[WARN] {task}: insufficient labels, skipping"); continue

        display = TASK_DISPLAY.get(task, task)
        n_pos   = int((y_all[valid] == 1).sum())
        n_unl   = int((y_all[valid] == 0).sum())
        print(f"\n[{display}]  |P|={n_pos}  |U|={n_unl}")

        # Use all valid genes for training; score ALL genes for ranking
        Xb = Xb_all[valid]; Xa = Xa_all[valid]; Xe = Xe_all[valid]
        y  = y_all[valid].astype(int)

        # Full-genome arrays for scoring (impute/scale separately below)
        Xb_full, Xa_full, Xe_full = Xb_all.copy(), Xa_all.copy(), Xe_all.copy()

        # Preprocess training set
        Xb_tr, Xa_tr, Xe_tr = preprocess_full(Xb, Xa, Xe, args.emb_pca_dim, args.seed)

        # Preprocess full genome using same transforms
        # (Re-fit on full set since we're not holding out anything)
        Xb_sc, Xa_sc, Xe_sc = preprocess_full(
            Xb_full, Xa_full, Xe_full, args.emb_pca_dim, args.seed
        )

        # Build and train
        model = TargetSage(
            d_bio=Xb_tr.shape[1], d_attr=Xa_tr.shape[1], d_emb=Xe_tr.shape[1],
            d_latent=args.d_latent, head_h=args.head_h, dropout=args.dropout,
            fusion=args.fusion,
        ).to(args.device)

        warmup(model, Xb_tr, Xa_tr, Xe_tr, y,
               args.warmup_epochs, args.lr, args.batch_size, args.device)

        prob_tr = predict(model, Xb_tr, Xa_tr, Xe_tr, args.device)
        pi_data = estimate_prior_en(prob_tr, y, args.seed, args.calib_frac, args.pi_cap)
        pi_llm  = prior_map.get(display, np.nan)
        pi_used = (
            float(args.alpha * pi_data + (1 - args.alpha) * pi_llm)
            if not np.isnan(pi_llm) else pi_data
        )
        pi_used = float(np.clip(pi_used, float(np.mean(y == 1)), args.pi_cap))
        print(f"  pi_data={pi_data:.4f}  pi_llm={pi_llm if not np.isnan(pi_llm) else 'N/A'}  "
              f"pi_used={pi_used:.4f}")

        train_nnpu(model, Xb_tr, Xa_tr, Xe_tr, y,
                   pi=pi_used, epochs=args.nnpu_epochs,
                   lr=args.lr, batch_size=args.batch_size,
                   beta=args.beta, device=args.device)

        # Score the full genome
        scores = predict(model, Xb_sc, Xa_sc, Xe_sc, args.device)

        # Build output DataFrame
        label_col = np.full(len(genes_all), np.nan)
        label_col[valid] = y_all[valid]

        ranking = pd.DataFrame({
            "Gene_Symbol":     genes_all,
            "targetsage_score": scores,
            "label":           label_col,
        }).sort_values("targetsage_score", ascending=False).reset_index(drop=True)
        ranking.index += 1
        ranking.index.name = "rank"

        # Save full ranking
        task_short = task.replace("task_", "")
        out_path   = os.path.join(outdir, f"{task_short}_ranking.csv")
        ranking.to_csv(out_path)
        print(f"  Saved full ranking ({len(ranking)} genes) -> {out_path}")

        # Optionally save top-K
        if args.top_k > 0:
            topk_path = os.path.join(outdir, f"{task_short}_top{args.top_k}.csv")
            ranking.head(args.top_k).to_csv(topk_path)
            n_known = int(ranking.head(args.top_k)["label"].eq(1).sum())
            print(f"  Top-{args.top_k}: {n_known} known positives  -> {topk_path}")

        # Optionally save model
        if args.save_model:
            ckpt_path = os.path.join(outdir, f"{task_short}_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Model checkpoint -> {ckpt_path}")

    print(f"\n[DONE] All outputs saved to {outdir}/")


if __name__ == "__main__":
    main()
