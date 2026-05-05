#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- IMPORTANT: set thread limits BEFORE importing numpy/sklearn ----
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Further limit threadpools if available (safe optional)
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
except Exception:
    pass

# Make repo root importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from targetsage import (
    TargetSage, nnpu_loss,
    load_bio_features, load_llm_scores, load_llm_embeddings,
    load_labels, build_feature_matrix, get_feature_arrays,
)

# -------------------------
# Utils
# -------------------------
def normalize_tdl(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    mapping = {"tclin": "Tclin", "tchem": "Tchem", "tbio": "Tbio", "tdark": "Tdark"}
    return mapping.get(s, str(x).strip())

@torch.no_grad()
def ts_predict_logits(model, Xb, Xa, Xe, device, batch=4096):
    """Return raw logits (more stable for ranking than sigmoid probs)."""
    model.eval()
    out = []
    for i in range(0, len(Xb), batch):
        lp, _, _ = model.forward_logits(
            torch.from_numpy(Xb[i:i+batch]).to(device),
            torch.from_numpy(Xa[i:i+batch]).to(device),
            torch.from_numpy(Xe[i:i+batch]).to(device),
        )
        out.append(lp.detach().cpu().numpy())
    return np.concatenate(out)

def focal_bce_with_logits(logits, targets, gamma=2.0, pos_weight=None):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = (1 - pt).pow(gamma)
    return (w * bce).mean()

def train_supervised(model, Xb, Xa, Xe, y, epochs, lr, batch_size, device,
                     use_focal=False, focal_gamma=2.0):
    """Pure supervised training (recommended for this old-TDL-defined label task)."""
    if epochs <= 0:
        return
    y = y.astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 10 or n_neg < 10:
        return

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    pw = min(n_neg / max(n_pos, 1), 20.0)          # clip pos_weight
    pos_weight = torch.tensor([pw], device=device)

    for _ in range(epochs):
        perm = np.random.permutation(len(y))        # reshuffle each epoch
        for i in range(0, len(y), batch_size):
            idx = perm[i:i+batch_size]
            lp, _, _ = model.forward_logits(
                torch.from_numpy(Xb[idx]).to(device),
                torch.from_numpy(Xa[idx]).to(device),
                torch.from_numpy(Xe[idx]).to(device),
            )
            tgt = torch.from_numpy(y[idx].astype(np.float32)).to(device)

            opt.zero_grad()
            if use_focal:
                loss = focal_bce_with_logits(lp, tgt, gamma=focal_gamma, pos_weight=pos_weight)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lp, tgt, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

def train_nnpu_semantic(model, Xb, Xa, Xe, y, pi, epochs, lr, batch_size, beta, device):
    """nnPU (optional) with semantic-guided reweighting."""
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
        sim_u = ((h_u_norm * centroid).sum(1) + 1.0) / 2.0
        w_u = torch.clamp(beta * prob[idx_u] + (1 - beta) * sim_u, 0, 1).cpu().numpy()

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
            w_batch = torch.from_numpy(w_u[[u_map[int(g)] for g in bu]]).to(device).float()
            loss = nnpu_loss(lp_p, lp_u, pi=pi, w_u=w_batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

def preprocess_three_streams(Xb, Xa, Xe, emb_pca_dim, seed):
    imp_b = SimpleImputer(strategy="median")
    imp_a = SimpleImputer(strategy="median")
    imp_e = SimpleImputer(strategy="median")
    Xb = imp_b.fit_transform(Xb)
    Xa = imp_a.fit_transform(Xa)
    Xe = imp_e.fit_transform(Xe)

    if emb_pca_dim > 0:
        k = min(emb_pca_dim, Xe.shape[1])
        Xe = PCA(n_components=k, random_state=seed).fit_transform(Xe)

    sc_b = StandardScaler(); sc_a = StandardScaler(); sc_e = StandardScaler()
    Xb = sc_b.fit_transform(Xb)
    Xa = sc_a.fit_transform(Xa)
    Xe = sc_e.fit_transform(Xe)
    return Xb, Xa, Xe

def preprocess_concat(Xb, Xa, Xe, emb_pca_dim, seed):
    imp_b = SimpleImputer(strategy="median")
    imp_a = SimpleImputer(strategy="median")
    imp_e = SimpleImputer(strategy="median")
    Xb = imp_b.fit_transform(Xb)
    Xa = imp_a.fit_transform(Xa)
    Xe = imp_e.fit_transform(Xe)

    if emb_pca_dim > 0:
        k = min(emb_pca_dim, Xe.shape[1])
        Xe = PCA(n_components=k, random_state=seed).fit_transform(Xe)

    X = np.concatenate([Xb, Xa, Xe], axis=1)
    X = StandardScaler().fit_transform(X)
    return X

def get_ranking_df(genes, scores, old_tdl, new_tdl):
    df = pd.DataFrame({
        "Gene_Symbol": genes,
        "score": scores.astype(float),
        "idgTDL_old": old_tdl,
        "idgTDL_new": new_tdl,
    }).sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df

def eval_upgrade_enrichment(ranking_df, upgraded_genes, top_fracs=(0.01, 0.05, 0.10)):
    N = len(ranking_df)
    gene_to_rank = {g: int(r) for r, g in zip(ranking_df.index.values, ranking_df["Gene_Symbol"].values)}
    ranks = [gene_to_rank[g] for g in upgraded_genes if g in gene_to_rank]

    if len(ranks) == 0:
        return {
            "n_upgraded": len(upgraded_genes),
            "n_found_in_genome": 0,
            "mean_percentile": np.nan,
            "median_percentile": np.nan,
            **{f"recall_top{int(fr*100)}pct": np.nan for fr in top_fracs},
        }

    ranks = np.array(ranks, dtype=float)
    percentiles = ranks / float(N)
    out = {
        "n_upgraded": len(upgraded_genes),
        "n_found_in_genome": int(len(ranks)),
        "mean_percentile": float(np.mean(percentiles)),
        "median_percentile": float(np.median(percentiles)),
    }
    for fr in top_fracs:
        k = max(1, int(fr * N))
        out[f"recall_top{int(fr*100)}pct"] = float(np.mean(ranks <= k))
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio", default="data/gene_features.tsv")
    ap.add_argument("--scores", default="data/features_llm_structured_scores.csv")
    ap.add_argument("--embeddings", default="data/features_llm_embedding.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results")

    # Use pre-computed TargetSage inference rankings instead of training from scratch
    ap.add_argument("--ts_inference_dir", default=None,
                    help="Directory with TargetSage inference CSVs (e.g. results/inference_YYYYMMDD). "
                         "If given, uses pharos_tclin_vs_others_ranking.csv etc. "
                         "instead of training TargetSage from scratch.")
    ap.add_argument("--ts_inference_map", nargs="*", default=None,
                    help="Pairs of task_key=csv_filename to map tasks to inference files.")

    ap.add_argument("--emb_pca_dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    # TargetSage params
    ap.add_argument("--d_latent", type=int, default=256)
    ap.add_argument("--head_h", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat", "sum"])
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--nnpu_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pi_cap", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # TargetSage training mode
    ap.add_argument("--ts_mode", default="supervised", choices=["supervised", "pu"])
    ap.add_argument("--ts_use_focal", action="store_true")
    ap.add_argument("--ts_focal_gamma", type=float, default=2.0)

    # NEW: scoring control
    ap.add_argument("--ts_score", default="logits", choices=["logits", "calibrated_prob"],
                    help="Use raw logits for ranking (recommended) or calibrated_prob (Platt scaling on train logits).")
    ap.add_argument("--ts_calibrate", action="store_true",
                    help="Alias for --ts_score calibrated_prob (kept for convenience).")

    # Ensemble
    ap.add_argument("--ensemble_seeds", type=int, nargs="+", default=None,
                    help="Train TargetSage with multiple seeds and average logits. "
                         "E.g. --ensemble_seeds 0 1 2 3 4")

    # Baselines
    ap.add_argument("--run_svm", action="store_true")
    ap.add_argument("--svm_cache_size", type=float, default=1024.0)
    ap.add_argument("--skip_knn", action="store_true")

    ap.add_argument("--skip_ensemble", action="store_true",
                    help="Skip ensemble baselines (RF, GB)")

    # Training subset
    ap.add_argument("--train_subset", default="HL", choices=["all", "HL"],
                    help="HL recommended: only {Tclin,Tchem,Tbio,Tdark} used for training.")

    args = ap.parse_args()
    if args.ts_calibrate:
        args.ts_score = "calibrated_prob"

    # Default inference mapping: task_key -> inference CSV filename
    ts_inference_map = {
        "upgrade_to_Tclin": "pharos_tclin_vs_others_ranking.csv",
        "upgrade_to_TclinOrTchem": "pharos_tclin_tchem_vs_others_ranking.csv",
    }
    if args.ts_inference_map:
        for pair in args.ts_inference_map:
            k, v = pair.split("=", 1)
            ts_inference_map[k] = v

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"upgrade_eval_{ts}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("[INFO] Loading features ...")
    bio_df = load_bio_features(args.bio)
    score_df = load_llm_scores(args.scores)
    emb_df = load_llm_embeddings(args.embeddings)

    # task labels only
    label_df = load_labels(args.labels)

    print("[INFO] Loading raw labels for TDL ...")
    labels_raw = pd.read_csv(args.labels, sep="\t", dtype=str)
    need_cols = ["Gene_Symbol", "idgTDL_old", "idgTDL_new"]
    missing = [c for c in need_cols if c not in labels_raw.columns]
    if missing:
        raise ValueError(f"Raw labels file missing columns: {missing}. Got: {list(labels_raw.columns)[:30]}")
    tdl_map_old = dict(zip(labels_raw["Gene_Symbol"].astype(str), labels_raw["idgTDL_old"]))
    tdl_map_new = dict(zip(labels_raw["Gene_Symbol"].astype(str), labels_raw["idgTDL_new"]))

    print("[INFO] Building merged feature matrix ...")
    merged = build_feature_matrix(bio_df, score_df, emb_df, label_df)
    Xb_all, Xa_all, Xe_all, genes_all = get_feature_arrays(merged)
    genes_all = np.array(genes_all, dtype=object)

    old_tdl = np.array([normalize_tdl(tdl_map_old.get(str(g), np.nan)) for g in genes_all], dtype=object)
    new_tdl = np.array([normalize_tdl(tdl_map_new.get(str(g), np.nan)) for g in genes_all], dtype=object)
    valid_tdl = (~pd.isna(old_tdl)) & (~pd.isna(new_tdl))

    # Truth sets from NEW
    upgraded_A = set(genes_all[
        valid_tdl & np.isin(old_tdl, ["Tchem", "Tbio", "Tdark"]) & (new_tdl == "Tclin")
    ])
    upgraded_B = set(genes_all[
        valid_tdl & np.isin(old_tdl, ["Tbio", "Tdark"]) & np.isin(new_tdl, ["Tclin", "Tchem"])
    ])

    tasks = {
        "upgrade_to_Tclin": upgraded_A,
        "upgrade_to_TclinOrTchem": upgraded_B,
    }

    # Training labels from OLD
    y_task = {
        "upgrade_to_Tclin": (old_tdl == "Tclin").astype(int),
        "upgrade_to_TclinOrTchem": np.isin(old_tdl, ["Tclin", "Tchem"]).astype(int),
    }

    print("[INFO] Preprocessing features (impute/scale/PCA) ...")
    Xb_sc, Xa_sc, Xe_sc = preprocess_three_streams(Xb_all, Xa_all, Xe_all, args.emb_pca_dim, args.seed)
    # Baselines use bio features only; TargetSage's advantage is multi-modal fusion
    X_baseline = SimpleImputer(strategy="median").fit_transform(Xb_all)
    X_baseline = StandardScaler().fit_transform(X_baseline)

    def fit_and_score_baselines(X_train, y_train, X_score):
        models = {
            "LR": LogisticRegression(max_iter=5000, n_jobs=1),
            "NB": GaussianNB(),
            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=args.seed),
        }
        if not args.skip_knn:
            models["KNN"] = KNeighborsClassifier(n_neighbors=10, n_jobs=1)

        if args.run_svm:
            models["SVM"] = SVC(
                kernel="rbf", probability=True,
                cache_size=args.svm_cache_size
            )

        if not args.skip_ensemble:
            models["RF"] = RandomForestClassifier(
                n_estimators=100, random_state=args.seed, n_jobs=1,
            )
            models["GB"] = GradientBoostingClassifier(subsample=0.5, random_state=args.seed)

        scores = {}
        for name, clf in models.items():
            print(f"    [BASELINE] fitting {name} ...")
            clf.fit(X_train, y_train)
            if hasattr(clf, "predict_proba"):
                s = clf.predict_proba(X_score)[:, 1]
            else:
                raw = clf.decision_function(X_score)
                s = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
            scores[name] = s
        return scores

    def _train_single_targetsage(Xb_train, Xa_train, Xe_train, y_train,
                                  Xb_score, Xa_score, Xe_score, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = TargetSage(
            d_bio=Xb_train.shape[1], d_attr=Xa_train.shape[1], d_emb=Xe_train.shape[1],
            d_latent=args.d_latent, head_h=args.head_h, dropout=args.dropout,
            fusion=args.fusion,
        ).to(args.device)

        if args.ts_mode == "supervised":
            sup_epochs = max(args.warmup_epochs, args.nnpu_epochs)
            train_supervised(
                model, Xb_train, Xa_train, Xe_train, y_train,
                epochs=sup_epochs, lr=args.lr, batch_size=args.batch_size,
                device=args.device, use_focal=args.ts_use_focal, focal_gamma=args.ts_focal_gamma
            )
        else:
            # PU mode: warmup supervised then nnPU
            pi_used = float(np.clip(np.mean(y_train == 1), float(np.mean(y_train == 1)), args.pi_cap))
            train_supervised(
                model, Xb_train, Xa_train, Xe_train, y_train,
                epochs=args.warmup_epochs, lr=args.lr, batch_size=args.batch_size,
                device=args.device, use_focal=False
            )
            train_nnpu_semantic(
                model, Xb_train, Xa_train, Xe_train, y_train,
                pi=pi_used, epochs=args.nnpu_epochs, lr=args.lr,
                batch_size=args.batch_size, beta=args.beta, device=args.device
            )

        logit_tr = ts_predict_logits(model, Xb_train, Xa_train, Xe_train, args.device)
        logit_sc = ts_predict_logits(model, Xb_score, Xa_score, Xe_score, args.device)
        return logit_tr, logit_sc

    def fit_and_score_targetsage(Xb_train, Xa_train, Xe_train, y_train,
                                 Xb_score, Xa_score, Xe_score):
        seeds = args.ensemble_seeds if args.ensemble_seeds else [args.seed]
        all_logit_tr, all_logit_sc = [], []
        for s in seeds:
            print(f"      [TargetSage] training seed={s} ...")
            ltr, lsc = _train_single_targetsage(
                Xb_train, Xa_train, Xe_train, y_train,
                Xb_score, Xa_score, Xe_score, seed=s
            )
            all_logit_tr.append(ltr)
            all_logit_sc.append(lsc)

        # average logits across ensemble members
        logit_tr = np.mean(all_logit_tr, axis=0)
        logit_sc = np.mean(all_logit_sc, axis=0)

        if args.ts_score == "logits":
            return logit_sc

        # calibrated_prob: Platt scaling on averaged logits
        calib = LogisticRegression(max_iter=2000, class_weight="balanced")
        calib.fit(logit_tr.reshape(-1, 1), y_train.astype(int))
        prob_sc = calib.predict_proba(logit_sc.reshape(-1, 1))[:, 1]
        return prob_sc

    summary_rows = []

    for task_name, upgraded_set in tasks.items():
        y_full = y_task[task_name].astype(int)

        train_mask = ~pd.isna(old_tdl)
        if args.train_subset == "HL":
            train_mask = train_mask & np.isin(old_tdl, ["Tclin", "Tchem", "Tbio", "Tdark"])

        y_train = y_full[train_mask].astype(int)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        print(f"\n[INFO] Task={task_name} train_subset={args.train_subset} |P|={n_pos} |N|={n_neg} upgraded={len(upgraded_set)}")

        if n_pos == 0 or n_neg == 0:
            print(f"[WARN] Degenerate y in training for {task_name}. Skipping.")
            continue

        Xc_train, Xc_score = X_baseline[train_mask], X_baseline
        Xb_train, Xa_train, Xe_train = Xb_sc[train_mask], Xa_sc[train_mask], Xe_sc[train_mask]
        Xb_score, Xa_score, Xe_score = Xb_sc, Xa_sc, Xe_sc

        baseline_scores = fit_and_score_baselines(Xc_train, y_train, Xc_score)

        # TargetSage: use pre-computed inference if available
        if args.ts_inference_dir and ts_inference_map.get(task_name):
            inf_csv = os.path.join(args.ts_inference_dir, ts_inference_map[task_name])
            print(f"    [TargetSage] using pre-computed inference: {inf_csv}")
            inf_df = pd.read_csv(inf_csv)
            # build score array aligned with genes_all
            g2s = dict(zip(inf_df["Gene_Symbol"].astype(str),
                           inf_df["targetsage_score"].astype(float)))
            ts_scores = np.array([g2s.get(str(g), 0.0) for g in genes_all])
        else:
            print(f"    [TargetSage] mode={args.ts_mode} score={args.ts_score} focal={args.ts_use_focal} ...")
            ts_scores = fit_and_score_targetsage(Xb_train, Xa_train, Xe_train, y_train,
                                                 Xb_score, Xa_score, Xe_score)

        df_ts = get_ranking_df(genes_all, ts_scores, old_tdl, new_tdl)
        df_ts.to_csv(os.path.join(outdir, f"{task_name}__TargetSage.csv"))
        met_ts = eval_upgrade_enrichment(df_ts, upgraded_set)
        summary_rows.append({"task": task_name, "method": "TargetSage", **met_ts})

        for m, s in baseline_scores.items():
            df_m = get_ranking_df(genes_all, s, old_tdl, new_tdl)
            df_m.to_csv(os.path.join(outdir, f"{task_name}__{m}.csv"))
            met = eval_upgrade_enrichment(df_m, upgraded_set)
            summary_rows.append({"task": task_name, "method": m, **met})

        pd.DataFrame({"Gene_Symbol": sorted(list(upgraded_set))}).to_csv(
            os.path.join(outdir, f"{task_name}__upgraded_genes.csv"),
            index=False
        )

    summary = pd.DataFrame(summary_rows)
    if len(summary) == 0:
        print("[ERROR] No results produced.")
        return

    summary = summary.sort_values(["task", "mean_percentile"], ascending=[True, True])
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print(f"\n[DONE] saved to: {outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()