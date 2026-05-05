#!/usr/bin/env python3
"""Fast CV evaluation using SAVED LLM attrs from training.
No LLM regeneration -- load llm_attrs_subset.csv from each task's output dir.
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)


TASK_DISPLAY = {
    "task_pharos_tclin_vs_others": ("Clinical Targets", None, "results/rl_grpo_v3"),
    "task_pharos_tclin_tchem_vs_others": ("Clinical & Chemical", "pharos_tclin_tchem", "results/grpo_multitask"),
    "task_triage_tier1_vs_others": ("Top-Tier", "triage_tier1", "results/grpo_multitask"),
    "task_triage_tier12_vs_others": ("High-Confidence", "triage_tier12", "results/grpo_multitask"),
    "task_cancer_druggability": ("Cancer-Relevant", "cancer_druggability", "results/grpo_multitask"),
    "task_cancer_type_specific_target_prioritization": ("Type-Specific", "cancer_type_specific", "results/grpo_multitask"),
    "task_pan_cancer_target_prioritization": ("Pan-Cancer", "pan_cancer", "results/grpo_multitask"),
    "task_T1_targets_only": ("T1 Cancer", "T1", "results/grpo_multitask"),
    "task_T1_T2_targets": ("T1-T2 Cancer", "T1_T2", "results/grpo_multitask"),
    "task_T1_T2_T3_targets": ("T1-T3 Cancer", "T1_T2_T3", "results/grpo_multitask"),
    "task_sm_bucket1_vs_others": ("SM (Appr.)", "sm_bucket1", "results/grpo_multitask"),
    "task_sm_bucket123_vs_others": ("SM (Clin+)", "sm_bucket123", "results/grpo_multitask"),
    "task_ab_bucket1_vs_others": ("Ab (Appr.)", "ab_bucket1", "results/grpo_multitask"),
    "task_ab_bucket123_vs_others": ("Ab (Clin+)", "ab_bucket123", "results/grpo_multitask"),
    "task_protac_bucket1234_vs_others": ("PROTAC", "protac", "results/grpo_multitask"),
}


def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


def cv_adj_f1(X, y, n_splits=5, seed=42):
    if y.sum() < n_splits or (y == 0).sum() < n_splits:
        # Not enough samples; use 80/20 split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sc = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                solver="lbfgs", random_state=seed)
        lr.fit(sc.transform(X_tr), y_tr)
        prob = lr.predict_proba(sc.transform(X_te))[:, 1]
        return adjusted_f1(prob, y_te) * 100, 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                solver="lbfgs", random_state=seed)
        lr.fit(sc.transform(X[tr_idx]), y[tr_idx])
        prob = lr.predict_proba(sc.transform(X[te_idx]))[:, 1]
        scores.append(adjusted_f1(prob, y[te_idx]))
    return float(np.mean(scores) * 100), float(np.std(scores) * 100)


def build_subset(task, genes_all, bio_all, labels_df, n_subset=1000, seed=42):
    """Reproduce the same subset as in training."""
    y_all = labels_df.set_index("Gene_Symbol").reindex(genes_all)[task].values
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    np.random.seed(seed)
    target_pos = min(len(pos_idx), max(int(n_subset * 0.7), 200))
    target_neg = min(len(neg_idx), n_subset - target_pos)
    if target_pos < len(pos_idx):
        pos_sample = np.random.choice(pos_idx, size=target_pos, replace=False)
    else:
        pos_sample = pos_idx
    neg_sample = np.random.choice(neg_idx, size=target_neg, replace=False)
    subset_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(subset_idx)
    subset_genes = [genes_all[i] for i in subset_idx]
    subset_bio = bio_all[subset_idx]
    subset_y = y_all[subset_idx].astype(int)
    return subset_genes, subset_bio, subset_y


def main():
    print("[DATA] Loading...")
    summaries = pd.read_csv("data/gene_summaries.tsv", sep="\t")
    labels_df = pd.read_csv("data/gene_labels.tsv", sep="\t")
    bio_df = pd.read_csv("data/gene_features.tsv", sep="\t").set_index("Gene_Symbol")
    bio_df = bio_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    agent_df = pd.read_csv("results/agent_tools/agent_attributes_filtered.csv")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))

    genes_all = sorted(
        set(g2s.keys()) & set(labels_df["Gene_Symbol"].astype(str))
        & set(bio_df.index) & set(agent_df["Gene_Symbol"].astype(str))
    )
    bio_all = bio_df.reindex(genes_all).values.astype(float)
    bio_all = np.nan_to_num(bio_all, nan=0.0)

    results = []
    for task, (display, short, parent) in TASK_DISPLAY.items():
        if short is None:
            attrs_path = f"{parent}/llm_attrs_subset.csv"
        else:
            attrs_path = f"{parent}/{short}/llm_attrs_subset.csv"

        if not os.path.exists(attrs_path):
            print(f"[SKIP] {display}: no saved attrs at {attrs_path}")
            continue

        subset_genes, subset_bio, subset_y = build_subset(task, genes_all, bio_all, labels_df)

        # Load saved LLM attrs
        llm_df = pd.read_csv(attrs_path, index_col=0)
        # Align to subset_genes
        llm_mat = llm_df.reindex(subset_genes).fillna(0.5).values.astype(float)

        # Bio-only CV
        bio_mean, bio_std = cv_adj_f1(subset_bio, subset_y)
        # LLM-only CV
        llm_mean, llm_std = cv_adj_f1(llm_mat, subset_y)
        # Fused CV
        fused_X = np.concatenate([subset_bio, llm_mat], axis=1)
        fused_mean, fused_std = cv_adj_f1(fused_X, subset_y)

        delta = fused_mean - bio_mean
        print(f"{display:<25} pos={subset_y.sum():>4}  bio={bio_mean:>6.2f}±{bio_std:>4.2f}  "
              f"llm={llm_mean:>6.2f}±{llm_std:>4.2f}  fused={fused_mean:>6.2f}±{fused_std:>4.2f}  Δ={delta:+.2f}")

        results.append({
            "task": task, "display": display,
            "n_pos": int(subset_y.sum()), "n_neg": int(len(subset_y) - subset_y.sum()),
            "bio_cv_mean": bio_mean, "bio_cv_std": bio_std,
            "llm_cv_mean": llm_mean, "llm_cv_std": llm_std,
            "fused_cv_mean": fused_mean, "fused_cv_std": fused_std,
            "improvement": delta,
        })

    df = pd.DataFrame(results)
    out_csv = "results/grpo_multitask/cv_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "="*120)
    print(f"{'Task':<25} {'n_pos':>6} {'Bio CV':>14} {'LLM CV':>14} {'Fused CV':>14} {'Δ(Fused-Bio)':>14}")
    print("="*120)
    for _, r in df.iterrows():
        print(f"{r['display']:<25} {r['n_pos']:>6} "
              f"{r['bio_cv_mean']:>7.2f} ±{r['bio_cv_std']:>4.2f}  "
              f"{r['llm_cv_mean']:>7.2f} ±{r['llm_cv_std']:>4.2f}  "
              f"{r['fused_cv_mean']:>7.2f} ±{r['fused_cv_std']:>4.2f}  "
              f"{r['improvement']:>+12.2f}%")
    print("="*120)
    print(f"{'MEAN':<25} {'':<6} "
          f"{df['bio_cv_mean'].mean():>12.2f}    "
          f"{df['llm_cv_mean'].mean():>12.2f}    "
          f"{df['fused_cv_mean'].mean():>12.2f}    "
          f"{df['improvement'].mean():>+12.2f}%")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
