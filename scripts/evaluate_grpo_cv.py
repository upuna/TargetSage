#!/usr/bin/env python3
"""Re-evaluate GRPO results using 5-fold CV for rigorous reporting."""
import os, sys, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/home/zihend1/Genesis/TargetSage2/scripts")
os.chdir("/home/zihend1/Genesis/TargetSage2")

from rl_grpo_v3 import ATTR_VOCAB, make_prompt, build_gene_profile, generate_batch, extract_attrs, attrs_to_vec

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


TASK_DISPLAY = {
    "task_pharos_tclin_vs_others": ("Clinical Targets", None),  # v3 dir
    "task_pharos_tclin_tchem_vs_others": ("Clinical & Chemical", "pharos_tclin_tchem"),
    "task_triage_tier1_vs_others": ("Top-Tier", "triage_tier1"),
    "task_triage_tier12_vs_others": ("High-Confidence", "triage_tier12"),
    "task_cancer_druggability": ("Cancer-Relevant", "cancer_druggability"),
    "task_cancer_type_specific_target_prioritization": ("Type-Specific", "cancer_type_specific"),
    "task_pan_cancer_target_prioritization": ("Pan-Cancer", "pan_cancer"),
    "task_T1_targets_only": ("T1 Cancer", "T1"),
    "task_T1_T2_targets": ("T1-T2 Cancer", "T1_T2"),
    "task_T1_T2_T3_targets": ("T1-T3 Cancer", "T1_T2_T3"),
    "task_sm_bucket1_vs_others": ("SM (Appr.)", "sm_bucket1"),
    "task_sm_bucket123_vs_others": ("SM (Clin+)", "sm_bucket123"),
    "task_ab_bucket1_vs_others": ("Ab (Appr.)", "ab_bucket1"),
    "task_ab_bucket123_vs_others": ("Ab (Clin+)", "ab_bucket123"),
    "task_protac_bucket1234_vs_others": ("PROTAC", "protac"),
}


def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


def cv_adj_f1(X, y, n_splits=5, seed=42):
    """5-fold stratified CV Adj F1."""
    if y.sum() < n_splits or (y == 0).sum() < n_splits:
        # Not enough samples for CV; fall back to 80/20 split
        from sklearn.model_selection import train_test_split
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


def gen_attrs_for_subset(model, tokenizer, subset_genes, g2s, agent_attrs_dict, device, temperature=0.3):
    """Regenerate LLM attrs for each gene using the trained RL checkpoint."""
    llm_mat = np.full((len(subset_genes), len(ATTR_VOCAB)), 0.5)
    BATCH = 8
    for i in range(0, len(subset_genes), BATCH):
        batch = subset_genes[i:i+BATCH]
        prompts = [make_prompt(g, build_gene_profile(g, g2s, agent_attrs_dict), tokenizer) for g in batch]
        texts, _ = generate_batch(model, tokenizer, prompts, device, temperature=temperature, max_tokens=250)
        for bi, (g, t) in enumerate(zip(batch, texts)):
            attrs, _ = extract_attrs(t)
            if attrs:
                llm_mat[i+bi] = attrs_to_vec(attrs)
    return llm_mat


def main():
    # Load common data
    print("[DATA] Loading...")
    summaries = pd.read_csv("data/gene_summaries.tsv", sep="\t")
    labels_df = pd.read_csv("data/gene_labels.tsv", sep="\t")
    bio_df_raw = pd.read_csv("data/gene_features.tsv", sep="\t").set_index("Gene_Symbol")
    bio_df = bio_df_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    agent_df = pd.read_csv("results/agent_tools/agent_attributes_filtered.csv")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    agent_attrs_dict = agent_df.set_index("Gene_Symbol").to_dict("index")

    genes_all = sorted(
        set(g2s.keys()) & set(labels_df["Gene_Symbol"].astype(str))
        & set(bio_df.index) & set(agent_df["Gene_Symbol"].astype(str))
    )
    bio_all = bio_df.reindex(genes_all).values.astype(float)
    bio_all = np.nan_to_num(bio_all, nan=0.0)

    # Load base model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loading base {MODEL_ID}...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

    results = []
    for task, (display, short) in TASK_DISPLAY.items():
        print(f"\n=== {display} ({task}) ===")
        ckpt = "results/rl_grpo_v3/rl_best" if short is None else f"results/grpo_multitask/{short}/rl_best"
        if not os.path.isdir(ckpt):
            print(f"  [SKIP] no checkpoint at {ckpt}")
            continue

        subset_genes, subset_bio, subset_y = build_subset(task, genes_all, bio_all, labels_df)
        pos_n = subset_y.sum()
        neg_n = len(subset_y) - pos_n
        print(f"  Subset: {pos_n} pos / {neg_n} neg")

        # Bio-only CV Adj F1
        bio_mean, bio_std = cv_adj_f1(subset_bio, subset_y)

        # Load RL checkpoint and generate attrs
        print(f"  Loading {ckpt} and regenerating attrs...")
        model = PeftModel.from_pretrained(base, ckpt)
        model.eval()
        llm_mat = gen_attrs_for_subset(model, tokenizer, subset_genes, g2s, agent_attrs_dict, device)
        # Detach LoRA so next task can load fresh
        model = model.unload()

        # Fused CV Adj F1
        fused_X = np.concatenate([subset_bio, llm_mat], axis=1)
        fused_mean, fused_std = cv_adj_f1(fused_X, subset_y)

        delta = fused_mean - bio_mean
        print(f"  Bio-only CV:  {bio_mean:.2f} ± {bio_std:.2f}")
        print(f"  Bio+RL CV:    {fused_mean:.2f} ± {fused_std:.2f}")
        print(f"  Improvement:  {delta:+.2f}%")

        results.append({
            "task": task,
            "display": display,
            "n_pos": int(pos_n),
            "n_neg": int(neg_n),
            "bio_cv_mean": bio_mean,
            "bio_cv_std": bio_std,
            "fused_cv_mean": fused_mean,
            "fused_cv_std": fused_std,
            "improvement": delta,
        })

        # Save incrementally
        pd.DataFrame(results).to_csv("results/grpo_multitask/cv_summary.csv", index=False)

    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print(f"{'Task':<25} {'n_pos':>8} {'Bio CV':>18} {'Bio+RL CV':>18} {'Δ':>8}")
    print("="*100)
    for _, r in df.iterrows():
        print(f"{r['display']:<25} {r['n_pos']:>8} "
              f"{r['bio_cv_mean']:>10.2f} ± {r['bio_cv_std']:>4.2f}  "
              f"{r['fused_cv_mean']:>10.2f} ± {r['fused_cv_std']:>4.2f}  "
              f"{r['improvement']:>+7.2f}%")
    print("="*100)
    print(f"{'MEAN':<25} {'':<8} {df['bio_cv_mean'].mean():>10.2f}            {df['fused_cv_mean'].mean():>10.2f}            "
          f"{df['improvement'].mean():>+7.2f}%")


if __name__ == "__main__":
    main()
