#!/usr/bin/env python3
"""
Comprehensive ablation study for TargetSage.

Panel A: Feature modality  (bio-only, bio+attr, bio+emb, bio+attr+emb)
         — for TargetSage and 7 classical ML baselines
Panel B: Fusion operator   (gated, concat, sum, attn)
Panel C: Learning objective (nnPU, uPU, BCE)

Usage:
    cd /home/zihend1/Genesis/TargetSage2
    python scripts/run_ablation.py --panels A B C --seeds 0,1,2,3,4
    python scripts/run_ablation.py --panels B      --seeds 0,1,2,3,4
"""

import os, sys, json, argparse
from datetime import datetime
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

from targetsage import (
    TargetSage, nnpu_loss, adjusted_f1,
    TASKS, TASK_DISPLAY,
    load_bio_features, load_llm_scores, load_llm_embeddings,
    load_labels, load_prior_map,
    build_feature_matrix, get_feature_arrays,
)

# ---------------------------------------------------------------------------
# Classical ML training
# ---------------------------------------------------------------------------
CLASSICAL_MODELS = {
    "LR":  LogisticRegression(max_iter=3000, C=1.0),
    "SVM": SVC(probability=True, C=1.0),
    "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=1),
    "NB":  GaussianNB(),
    "RF":  RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1),
    "GB":  GradientBoostingClassifier(n_estimators=100, random_state=0),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0),
}

def run_classical(X_tr, X_te, y_tr, y_te, model_name):
    clf = clone(CLASSICAL_MODELS[model_name])
    clf.random_state = None  # will be set per seed externally
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    return adjusted_f1(y_te, prob)

# ---------------------------------------------------------------------------
# TargetSage helpers (re-used from train.py)
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
    if epochs <= 0: return
    y = y.astype(int)
    n_pos, n_unl = int((y==1).sum()), int((y==0).sum())
    if n_pos < 10 or n_unl < 10: return
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_unl/max(n_pos,1)], device=device))
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


def train_nnpu(model, Xb, Xa, Xe, y, pi, epochs, lr, batch_size, beta, device,
               non_negative=True):
    if epochs <= 0: return
    idx_p = np.where(y==1)[0]; idx_u = np.where(y==0)[0]
    if len(idx_p)<10 or len(idx_u)<10: return
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
        w_u      = torch.clamp(beta * prob[idx_u] + (1-beta) * sim_u, 0, 1).cpu().numpy()
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
                w_u[[u_map[int(g)] for g in bu]]).to(device).float()
            loss = nnpu_loss(lp_p, lp_u, pi=pi, w_u=w_batch, non_negative=non_negative)
            opt.zero_grad(); loss.backward(); opt.step()


def estimate_prior(prob_tr, y_tr, seed, pi_cap=0.10):
    idx_p = np.where(y_tr==1)[0]; idx_u = np.where(y_tr==0)[0]
    if len(idx_p)<5 or len(idx_u)<5:
        return float(np.mean(y_tr==1))
    rng     = np.random.RandomState(seed)
    cal_idx = rng.permutation(idx_p)[:max(1, int(0.3*len(idx_p)))]
    c       = float(np.clip(np.mean(prob_tr[cal_idx]), 1e-3, 1.0))
    pi_raw  = float(np.mean(prob_tr[idx_u]))
    return float(np.clip(pi_raw/c, float(np.mean(y_tr==1)), pi_cap))


def preprocess(Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te, seed, emb_pca=256):
    def imp(tr, te):
        m = SimpleImputer(strategy="median"); return m.fit_transform(tr), m.transform(te)
    def sc(tr, te):
        s = StandardScaler(); return s.fit_transform(tr), s.transform(te)
    Xb_tr, Xb_te = imp(Xb_tr, Xb_te); Xa_tr, Xa_te = imp(Xa_tr, Xa_te)
    Xe_tr, Xe_te = imp(Xe_tr, Xe_te)
    if emb_pca > 0 and Xe_tr.shape[1] > emb_pca:
        pca = PCA(n_components=emb_pca, random_state=seed)
        Xe_tr = pca.fit_transform(Xe_tr); Xe_te = pca.transform(Xe_te)
    Xb_tr, Xb_te = sc(Xb_tr, Xb_te); Xa_tr, Xa_te = sc(Xa_tr, Xa_te)
    Xe_tr, Xe_te = sc(Xe_tr, Xe_te)
    return Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te


def run_targetsage(Xb_tr, Xa_tr, Xe_tr, Xb_te, Xa_te, Xe_te, y_tr, y_te,
                   pi_used, fusion, objective, seed, device,
                   d_latent=256, head_h=512, dropout=0.2,
                   warmup_ep=10, nnpu_ep=25, lr=2e-4, batch=512, beta=0.6):
    model = TargetSage(
        d_bio=Xb_tr.shape[1], d_attr=Xa_tr.shape[1], d_emb=Xe_tr.shape[1],
        d_latent=d_latent, head_h=head_h, dropout=dropout, fusion=fusion,
    ).to(device)
    warmup(model, Xb_tr, Xa_tr, Xe_tr, y_tr, warmup_ep, lr, batch, device)
    if objective == "bce":
        pass  # warmup only = BCE
    else:
        non_neg = (objective == "nnpu")
        train_nnpu(model, Xb_tr, Xa_tr, Xe_tr, y_tr, pi=pi_used,
                   epochs=nnpu_ep, lr=lr, batch_size=batch, beta=beta,
                   device=device, non_negative=non_neg)
    prob = predict(model, Xb_te, Xa_te, Xe_te, device)
    return adjusted_f1(y_te, prob)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=["A","B","C"],
                    choices=["A","B","C"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--outdir", default="results/ablation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tasks",  default=",".join(TASKS))
    args = ap.parse_args()

    seeds     = [int(s) for s in args.seeds.split(",")]
    task_list = [t.strip() for t in args.tasks.split(",")]
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir    = f"{args.outdir}_{ts}"
    os.makedirs(outdir, exist_ok=True)
    print(f"[INFO] Output: {outdir}  panels={args.panels}  seeds={seeds}  device={args.device}")

    # ---- Load data ----
    print("[INFO] Loading data...")
    bio_df    = load_bio_features("data/gene_features.tsv")
    score_df  = load_llm_scores("data/features_llm_structured_scores.csv")
    emb_df    = load_llm_embeddings("data/features_llm_embedding.csv")
    label_df  = load_labels("data/gene_labels.tsv")
    prior_map = load_prior_map("data/prior_task_summary.csv")
    merged    = build_feature_matrix(bio_df, score_df, emb_df, label_df)
    Xb_all, Xa_all, Xe_all, genes_all = get_feature_arrays(merged)
    print(f"[INFO] Genes={len(merged)} bio={Xb_all.shape[1]} attr={Xa_all.shape[1]} emb={Xe_all.shape[1]}")

    rows = []

    def run_one_task(task):
        if task not in merged.columns: return []
        y_all = merged[task].to_numpy()
        valid = ~np.isnan(y_all)
        if valid.sum() < 50 or len(np.unique(y_all[valid])) < 2: return []
        Xb = Xb_all[valid]; Xa = Xa_all[valid]; Xe = Xe_all[valid]
        y  = y_all[valid].astype(int)
        display = TASK_DISPLAY.get(task, task)
        pi_llm  = prior_map.get(display, np.nan)
        local_rows = []

        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed)
            try:
                bc = np.bincount(y)
                strat = y if (len(bc)>=2 and bc.min()>=2) else None
            except Exception: strat = None

            splits = train_test_split(
                Xb, Xa, Xe, y, test_size=0.2, random_state=seed, stratify=strat)
            Xb_tr,Xb_te,Xa_tr,Xa_te,Xe_tr,Xe_te,y_tr,y_te = splits
            Xb_tr,Xa_tr,Xe_tr,Xb_te,Xa_te,Xe_te = preprocess(
                Xb_tr,Xa_tr,Xe_tr,Xb_te,Xa_te,Xe_te, seed)

            # ---------------------------------------------------------------
            # Panel A: feature modality
            # ---------------------------------------------------------------
            if "A" in args.panels:
                # Classical ML on different feature combos
                Xa_flat_tr = Xa_tr; Xa_flat_te = Xa_te
                Xe_flat_tr = Xe_tr; Xe_flat_te = Xe_te

                combos = {
                    "bio":          (Xb_tr, Xb_te),
                    "bio+attr":     (np.hstack([Xb_tr, Xa_flat_tr]), np.hstack([Xb_te, Xa_flat_te])),
                    "bio+emb":      (np.hstack([Xb_tr, Xe_flat_tr]), np.hstack([Xb_te, Xe_flat_te])),
                    "bio+attr+emb": (np.hstack([Xb_tr, Xa_flat_tr, Xe_flat_tr]),
                                     np.hstack([Xb_te, Xa_flat_te, Xe_flat_te])),
                }
                for combo_name, (Xtr_, Xte_) in combos.items():
                    for mname in CLASSICAL_MODELS:
                        try:
                            score = run_classical(Xtr_, Xte_, y_tr, y_te, mname)
                        except Exception as e:
                            score = float("nan")
                            print(f"  [WARN] classical {mname} {combo_name} {display}: {e}")
                        local_rows.append(dict(
                            panel="A", task=task, display=display, seed=seed,
                            condition=combo_name, method=mname, adj_f1=score))

                # TargetSage modality ablation (zero-out missing modalities)
                d_attr = Xa_tr.shape[1]; d_emb = Xe_tr.shape[1]
                zero_a_tr = np.zeros_like(Xa_tr); zero_a_te = np.zeros_like(Xa_te)
                zero_e_tr = np.zeros_like(Xe_tr); zero_e_te = np.zeros_like(Xe_te)

                ts_combos = {
                    "bio":          (Xb_tr, zero_a_tr, zero_e_tr, Xb_te, zero_a_te, zero_e_te),
                    "bio+attr":     (Xb_tr, Xa_tr,     zero_e_tr, Xb_te, Xa_te,     zero_e_te),
                    "bio+emb":      (Xb_tr, zero_a_tr, Xe_tr,     Xb_te, zero_a_te, Xe_te),
                    "bio+attr+emb": (Xb_tr, Xa_tr,     Xe_tr,     Xb_te, Xa_te,     Xe_te),
                }

                # build prior once
                model_tmp = TargetSage(d_bio=Xb_tr.shape[1], d_attr=d_attr, d_emb=d_emb,
                                       d_latent=256, head_h=512, dropout=0.2, fusion="gated"
                                       ).to(args.device)
                warmup(model_tmp, Xb_tr, Xa_tr, Xe_tr, y_tr, 5, 2e-4, 512, args.device)
                prob_tmp = predict(model_tmp, Xb_tr, Xa_tr, Xe_tr, args.device)
                pi_data  = estimate_prior(prob_tmp, y_tr, seed)
                pi_used  = (float(0.5*pi_data + 0.5*pi_llm) if not np.isnan(pi_llm)
                             else pi_data)
                pi_used  = float(np.clip(pi_used, float(np.mean(y_tr==1)), 0.10))
                del model_tmp

                for combo_name, (b_tr,a_tr,e_tr,b_te,a_te,e_te) in ts_combos.items():
                    try:
                        score = run_targetsage(b_tr,a_tr,e_tr,b_te,a_te,e_te,
                                               y_tr, y_te, pi_used,
                                               fusion="gated", objective="nnpu",
                                               seed=seed, device=args.device)
                    except Exception as ex:
                        score = float("nan")
                        print(f"  [WARN] TS modality {combo_name} {display}: {ex}")
                    local_rows.append(dict(
                        panel="A", task=task, display=display, seed=seed,
                        condition=combo_name, method="TargetSage", adj_f1=score))

            # ---------------------------------------------------------------
            # Shared prior for panels B & C
            # ---------------------------------------------------------------
            if "B" in args.panels or "C" in args.panels:
                model_tmp = TargetSage(d_bio=Xb_tr.shape[1], d_attr=Xa_tr.shape[1],
                                       d_emb=Xe_tr.shape[1], d_latent=256, head_h=512,
                                       dropout=0.2, fusion="gated").to(args.device)
                warmup(model_tmp, Xb_tr, Xa_tr, Xe_tr, y_tr, 5, 2e-4, 512, args.device)
                prob_tmp = predict(model_tmp, Xb_tr, Xa_tr, Xe_tr, args.device)
                pi_data  = estimate_prior(prob_tmp, y_tr, seed)
                pi_used  = (float(0.5*pi_data + 0.5*pi_llm) if not np.isnan(pi_llm)
                             else pi_data)
                pi_used  = float(np.clip(pi_used, float(np.mean(y_tr==1)), 0.10))
                del model_tmp

            # ---------------------------------------------------------------
            # Panel B: fusion operator
            # ---------------------------------------------------------------
            if "B" in args.panels:
                for fusion in ["gated", "attn", "concat", "sum"]:
                    try:
                        score = run_targetsage(Xb_tr,Xa_tr,Xe_tr,Xb_te,Xa_te,Xe_te,
                                               y_tr, y_te, pi_used,
                                               fusion=fusion, objective="nnpu",
                                               seed=seed, device=args.device)
                    except Exception as ex:
                        score = float("nan")
                        print(f"  [WARN] fusion {fusion} {display}: {ex}")
                    local_rows.append(dict(
                        panel="B", task=task, display=display, seed=seed,
                        condition=fusion, method="TargetSage", adj_f1=score))

            # ---------------------------------------------------------------
            # Panel C: learning objective
            # ---------------------------------------------------------------
            if "C" in args.panels:
                for objective in ["nnpu", "upu", "bce"]:
                    try:
                        score = run_targetsage(Xb_tr,Xa_tr,Xe_tr,Xb_te,Xa_te,Xe_te,
                                               y_tr, y_te, pi_used,
                                               fusion="gated", objective=objective,
                                               seed=seed, device=args.device)
                    except Exception as ex:
                        score = float("nan")
                        print(f"  [WARN] obj {objective} {display}: {ex}")
                    local_rows.append(dict(
                        panel="C", task=task, display=display, seed=seed,
                        condition=objective, method="TargetSage", adj_f1=score))

        return local_rows

    # Run all tasks
    for task in task_list:
        display = TASK_DISPLAY.get(task, task)
        print(f"  Task: {display}")
        task_rows = run_one_task(task)
        rows.extend(task_rows)
        # Save incrementally
        pd.DataFrame(rows).to_csv(os.path.join(outdir, "raw.csv"), index=False)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(os.path.join(outdir, "raw.csv"), index=False)

    # Aggregate
    agg = (raw_df.groupby(["panel","condition","method","display"])
           .agg(mean=("adj_f1","mean"), std=("adj_f1","std"), n=("seed","count"))
           .reset_index())
    agg.to_csv(os.path.join(outdir, "agg.csv"), index=False)

    # Print summary per panel
    for panel in ["A","B","C"]:
        sub = agg[agg.panel==panel]
        if sub.empty: continue
        print(f"\n{'='*70}\nPanel {panel}\n{'='*70}")
        if panel == "A":
            pivot = sub[sub.method=="TargetSage"].pivot_table(
                index="display", columns="condition", values="mean")
            print(pivot.to_string())
        else:
            pivot = sub.pivot_table(index="display", columns="condition", values="mean")
            print(pivot.to_string())

    print(f"\n[DONE] Results: {outdir}/")


if __name__ == "__main__":
    main()
