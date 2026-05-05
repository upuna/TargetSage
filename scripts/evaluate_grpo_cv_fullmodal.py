#!/usr/bin/env python3
"""Evaluate GRPO checkpoints with full three-modal features: bio + attr + emb."""

import json
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")

from rl_grpo_v3 import (  # noqa: E402
    ATTR_VOCAB,
    attrs_to_vec,
    build_gene_profile,
    extract_attrs,
    generate_batch,
    make_prompt,
)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


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


def adjusted_f1(probs: np.ndarray, y: np.ndarray) -> float:
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    r_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(r_soft ** 2 / max(p_bar, 1e-10))


def cv_adj_f1(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> Tuple[float, float]:
    if y.sum() < n_splits or (y == 0).sum() < n_splits:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sc = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed)
        lr.fit(sc.transform(X_tr), y_tr)
        prob = lr.predict_proba(sc.transform(X_te))[:, 1]
        return adjusted_f1(prob, y_te) * 100, 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed)
        lr.fit(sc.transform(X[tr_idx]), y[tr_idx])
        prob = lr.predict_proba(sc.transform(X[te_idx]))[:, 1]
        scores.append(adjusted_f1(prob, y[te_idx]))
    return float(np.mean(scores) * 100), float(np.std(scores) * 100)


def cv_auroc(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> Tuple[float, float]:
    if y.sum() < n_splits or (y == 0).sum() < n_splits:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        sc = StandardScaler().fit(X_tr)
        lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed)
        lr.fit(sc.transform(X_tr), y_tr)
        prob = lr.predict_proba(sc.transform(X_te))[:, 1]
        return float(roc_auc_score(y_te, prob)), 0.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in skf.split(X, y):
        sc = StandardScaler().fit(X[tr_idx])
        lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", solver="lbfgs", random_state=seed)
        lr.fit(sc.transform(X[tr_idx]), y[tr_idx])
        prob = lr.predict_proba(sc.transform(X[te_idx]))[:, 1]
        scores.append(roc_auc_score(y[te_idx], prob))
    return float(np.mean(scores)), float(np.std(scores))


def build_subset(task: str, genes_all: List[str], bio_all: np.ndarray, labels_df: pd.DataFrame,
                 n_subset: int = 1000, seed: int = 42):
    y_all = labels_df.set_index("Gene_Symbol").reindex(genes_all)[task].values
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    np.random.seed(seed)
    target_pos = min(len(pos_idx), max(int(n_subset * 0.7), 200))
    target_neg = min(len(neg_idx), n_subset - target_pos)
    pos_sample = np.random.choice(pos_idx, size=target_pos, replace=False) if target_pos < len(pos_idx) else pos_idx
    neg_sample = np.random.choice(neg_idx, size=target_neg, replace=False)
    subset_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(subset_idx)
    subset_genes = [genes_all[i] for i in subset_idx]
    subset_bio = bio_all[subset_idx]
    subset_y = y_all[subset_idx].astype(int)
    return subset_genes, subset_bio, subset_y


def reasoning_text_for_embedding(gene: str, profile: str, reasoning: str) -> str:
    reasoning = (reasoning or "").strip()
    if not reasoning:
        reasoning = "No explicit reasoning generated."
    return f"Gene: {gene}\nEvidence: {profile}\nReasoning: {reasoning}"


def embed_texts_with_model(model, tokenizer, texts, device, batch_size: int = 16) -> np.ndarray:
    """Mean-pool the last hidden state over non-pad tokens."""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vecs.append(pooled.float().cpu().numpy())
    return np.vstack(vecs).astype(np.float32)


def gen_attrs_and_embeddings_for_subset(model, tokenizer, subset_genes, g2s, agent_attrs_dict,
                                        cache_dir: str, device,
                                        temperature: float = 0.3, emb_pca_dim: int = 256):
    os.makedirs(cache_dir, exist_ok=True)
    attrs_csv = os.path.join(cache_dir, "llm_attrs_subset.csv")
    emb_npy = os.path.join(cache_dir, "llm_reasoning_emb_subset.npy")
    reasoning_jsonl = os.path.join(cache_dir, "reasoning_subset.jsonl")

    if os.path.exists(attrs_csv) and os.path.exists(emb_npy):
        llm_df = pd.read_csv(attrs_csv, index_col=0)
        llm_mat = llm_df.reindex(subset_genes).fillna(0.5).values.astype(float)
        emb_mat = np.load(emb_npy)
        return llm_mat, emb_mat

    if os.path.exists(attrs_csv) and os.path.exists(reasoning_jsonl):
        llm_df = pd.read_csv(attrs_csv, index_col=0)
        llm_mat = llm_df.reindex(subset_genes).fillna(0.5).values.astype(np.float32)
        reasoning_map = {}
        with open(reasoning_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                reasoning_map[rec["gene"]] = rec
        reasoning_records = [reasoning_map[g] for g in subset_genes]
    else:
        llm_mat = np.full((len(subset_genes), len(ATTR_VOCAB)), 0.5, dtype=np.float32)
        reasoning_records = []

        batch_size = 8
        for i in range(0, len(subset_genes), batch_size):
            batch = subset_genes[i:i + batch_size]
            profiles = [build_gene_profile(g, g2s, agent_attrs_dict) for g in batch]
            prompts = [make_prompt(g, p, tokenizer) for g, p in zip(batch, profiles)]
            texts, _ = generate_batch(model, tokenizer, prompts, device, temperature=temperature, max_tokens=250)
            for bi, (g, p, text) in enumerate(zip(batch, profiles, texts)):
                attrs, reasoning = extract_attrs(text)
                if attrs:
                    llm_mat[i + bi] = attrs_to_vec(attrs)
                reasoning_records.append({
                    "gene": g,
                    "profile": p,
                    "raw_text": text,
                    "reasoning": reasoning,
                })
            print(f"    Generated {min(i + batch_size, len(subset_genes))}/{len(subset_genes)} genes", flush=True)

        llm_df = pd.DataFrame(llm_mat, index=subset_genes, columns=ATTR_VOCAB)
        llm_df.to_csv(attrs_csv)
        with open(reasoning_jsonl, "w") as f:
            for rec in reasoning_records:
                f.write(json.dumps(rec) + "\n")

    # Encode "evidence + reasoning" text locally, then PCA to align with the main model.
    embed_texts = [
        reasoning_text_for_embedding(rec["gene"], rec["profile"], rec["reasoning"])
        for rec in reasoning_records
    ]
    print(f"    Encoding {len(embed_texts)} reasoning texts into local embeddings...", flush=True)
    emb_mat = embed_texts_with_model(model, tokenizer, embed_texts, device=device, batch_size=16)

    if emb_pca_dim > 0 and emb_mat.shape[1] > emb_pca_dim:
        pca = PCA(n_components=min(emb_pca_dim, emb_mat.shape[1]), random_state=42)
        emb_mat = pca.fit_transform(emb_mat).astype(np.float32)

    np.save(emb_npy, emb_mat)
    return llm_mat, emb_mat


def main():
    print("[DATA] Loading...")
    summaries = pd.read_csv("data/gene_summaries.tsv", sep="\t")
    labels_df = pd.read_csv("data/gene_labels.tsv", sep="\t")
    bio_df_raw = pd.read_csv("data/gene_features.tsv", sep="\t").set_index("Gene_Symbol")
    bio_df = bio_df_raw.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    agent_df = pd.read_csv("results/agent_tools/agent_attributes_filtered.csv")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    agent_attrs_dict = agent_df.set_index("Gene_Symbol").to_dict("index")

    genes_all = sorted(
        set(g2s.keys()) & set(labels_df["Gene_Symbol"].astype(str))
        & set(bio_df.index) & set(agent_df["Gene_Symbol"].astype(str))
    )
    bio_all = bio_df.reindex(genes_all).values.astype(float)
    bio_all = np.nan_to_num(bio_all, nan=0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[MODEL] Loading base {model_id}...")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )

    full_dir = "results/grpo_multitask_fullmodal"
    os.makedirs(full_dir, exist_ok=True)

    cv_rows = []
    auroc_rows = []
    for task, (display, short) in TASK_DISPLAY.items():
        print(f"\n=== {display} ({task}) ===")
        ckpt = "results/rl_grpo_v3/rl_best" if short is None else f"results/grpo_multitask/{short}/rl_best"
        if not os.path.isdir(ckpt):
            print(f"  [SKIP] no checkpoint at {ckpt}")
            continue

        subset_genes, subset_bio, subset_y = build_subset(task, genes_all, bio_all, labels_df)
        pos_n = int(subset_y.sum())
        neg_n = int(len(subset_y) - pos_n)
        print(f"  Subset: {pos_n} pos / {neg_n} neg")

        bio_mean, bio_std = cv_adj_f1(subset_bio, subset_y)
        bio_auc_mean, bio_auc_std = cv_auroc(subset_bio, subset_y)

        print(f"  Loading {ckpt} and regenerating attrs + reasoning embeddings...")
        model = PeftModel.from_pretrained(base, ckpt)
        model.eval()
        cache_dir = os.path.join(full_dir, short if short is not None else "clinical_targets")
        llm_mat, emb_mat = gen_attrs_and_embeddings_for_subset(
            model, tokenizer, subset_genes, g2s, agent_attrs_dict, cache_dir, device
        )
        model = model.unload()

        fused_X = np.concatenate([subset_bio, llm_mat, emb_mat], axis=1)
        fused_mean, fused_std = cv_adj_f1(fused_X, subset_y)
        fused_auc_mean, fused_auc_std = cv_auroc(fused_X, subset_y)

        delta = fused_mean - bio_mean
        delta_auc = fused_auc_mean - bio_auc_mean
        print(f"  Bio-only CV:    {bio_mean:.2f} ± {bio_std:.2f}")
        print(f"  Bio+Attr+Emb:   {fused_mean:.2f} ± {fused_std:.2f}")
        print(f"  Improvement:    {delta:+.2f}%")

        cv_rows.append({
            "task": task,
            "display": display,
            "n_pos": pos_n,
            "n_neg": neg_n,
            "bio_cv_mean": bio_mean,
            "bio_cv_std": bio_std,
            "full_cv_mean": fused_mean,
            "full_cv_std": fused_std,
            "improvement": delta,
        })
        auroc_rows.append({
            "task": task,
            "display": display,
            "bio_auroc_mean": bio_auc_mean,
            "bio_auroc_std": bio_auc_std,
            "full_auroc_mean": fused_auc_mean,
            "full_auroc_std": fused_auc_std,
            "improvement": delta_auc,
        })

        pd.DataFrame(cv_rows).to_csv(os.path.join(full_dir, "cv_summary_fullmodal.csv"), index=False)
        pd.DataFrame(auroc_rows).to_csv(os.path.join(full_dir, "auroc_summary_fullmodal.csv"), index=False)

    cv_df = pd.DataFrame(cv_rows)
    auroc_df = pd.DataFrame(auroc_rows)
    print("\n" + "=" * 110)
    print(f"{'Task':<25} {'n_pos':>6} {'Bio CV':>18} {'Bio+Attr+Emb CV':>20} {'Δ':>10}")
    print("=" * 110)
    for _, r in cv_df.iterrows():
        print(
            f"{r['display']:<25} {r['n_pos']:>6} "
            f"{r['bio_cv_mean']:>8.2f} ± {r['bio_cv_std']:>5.2f}  "
            f"{r['full_cv_mean']:>10.2f} ± {r['full_cv_std']:>5.2f}  "
            f"{r['improvement']:>+9.2f}%"
        )
    print("=" * 110)
    print(
        f"{'MEAN':<25} {'':<6} "
        f"{cv_df['bio_cv_mean'].mean():>8.2f}               "
        f"{cv_df['full_cv_mean'].mean():>10.2f}               "
        f"{cv_df['improvement'].mean():>+9.2f}%"
    )
    print(f"\nSaved: {os.path.join(full_dir, 'cv_summary_fullmodal.csv')}")
    print(f"Saved: {os.path.join(full_dir, 'auroc_summary_fullmodal.csv')}")


if __name__ == "__main__":
    main()
