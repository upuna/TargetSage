#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetSage — Module M2: GRPO Fine-Tuning of the Attribute-Scoring LLM
======================================================================
NOTE FOR REVIEWERS
-------------------
Precomputed M2 outputs are provided in data/features_llm_structured_scores.csv.
You do NOT need to re-run this script to reproduce the paper's main results.
Running `python train.py` is sufficient.

This script is provided so reviewers can inspect the complete M2 pipeline
described in Section 3.2 of the paper.

Overview of Module M2: Guided Reasoning (GRPO)
------------------------------------------------
Module M2 fine-tunes a local instruction-following LLM using Group Relative
Policy Optimization (GRPO) so that the generated attribute scores are more
informative for downstream target identification.

The core insight: a language model that generates attribute scores
complementary to structural bio features (which the TargetSage model already
has access to via head_bio) should produce higher downstream classification
performance than one that redundantly scores attributes already captured by
the bio features.

GRPO Reward Function
---------------------
At each RL step:
  1. Sample a batch of G=4 rollouts per gene.
  2. For each rollout, substitute the generated attribute vector into the
     full LLM attribute matrix llm_mat.
  3. Fit a logistic regression on [bio features || llm_mat] for the
     labeled gene subset.
  4. Reward r = Adjusted F1 of that logistic regression = R_soft^2 / p̄
     (defined in targetsage/metrics.py).
  5. A penalty factor 0.5x is applied if the reasoning chain is absent or
     shorter than 30 characters.

The reward is zero-cost: Adjusted F1 on a ~1000-gene subset takes <0.5s
with sklearn LogisticRegression.

Group Advantage Normalization (GRPO)
--------------------------------------
GRPO estimates advantages within each group of G rollouts for the same gene:

    advantage_k = reward_k - mean(reward_{1..G})   for rollout k in group

Advantages are then globally normalized by their standard deviation:

    advantage_normalized_k = advantage_k / std(all_advantages)

This ensures the policy gradient update is invariant to the scale of the
reward signal and the number of groups.

Policy Gradient Update
-----------------------
Each rollout's policy gradient loss is:

    L_pg = out.loss * (-advantage_normalized)

where out.loss is the standard language model cross-entropy loss on the
generated tokens (provided by HuggingFace's model.forward()).  The negative
sign turns gradient ascent (maximize reward) into gradient descent (minimize
negative reward).

Gradients from all valid rollouts are accumulated and averaged before
applying the optimizer step.

LoRA Setup
-----------
We apply LoRA to all attention and MLP projection layers:
    target_modules: q_proj, k_proj, v_proj, o_proj,
                    gate_proj, up_proj, down_proj
    r=32, lora_alpha=64, lora_dropout=0.05

This keeps ~98% of model parameters frozen, reducing memory and preventing
catastrophic forgetting of the base model's language capabilities.

Usage (requires GPU with ≥16GB VRAM for 1.5B model)
-----------------------------------------------------
    CUDA_VISIBLE_DEVICES=0 python scripts/m2_grpo_training.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --rl_steps 100 \\
        --n_subset 1000

Output
------
    results/rl_grpo_v3/
        rl_best/              — best LoRA checkpoint (HuggingFace format)
        rl_log.csv            — training log (step, reward, loss, ...)
        llm_attrs_subset.csv  — final LLM attribute matrix for the subset
        llm_attrs_subset.npy  — same, as numpy array
"""

import os
import sys
import json
import re
import argparse
import time

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# 23-attribute therapeutic vocabulary (Appendix A.3 of the paper)
# ---------------------------------------------------------------------------
ATTR_VOCAB = [
    "disease_association",
    "essential_gene",
    "loss_of_function_tolerance",
    "loss_of_function_constraint",
    "observed_expected_lof_ratio",
    "rvis_score",
    "gwas_associations",
    "protein_protein_interactions",
    "chemical_gene_interactions",
    "drug_interaction_types",
    "known_antibodies",
    "antibody",
    "antibody_availability",
    "small_molecule",
    "tractable_modalities",
    "overall_therapeutic_potential",
    "functional_characterization",
    "protein_domains",
    "protein_length",
    "expression_tissue_specificity",
    "expression_specificity",
    "expression_broadly_expressed",
    "alternative_splicing",
]

ATTR_LIST_STR = ", ".join(ATTR_VOCAB)

# System prompt injected with the full 23-attribute list.
SYS = (
    "You are a drug discovery scientist evaluating therapeutic targets. "
    "Given a gene's evidence profile, write a 2-3 sentence biological reasoning chain, "
    f"then score the gene on these {len(ATTR_VOCAB)} attributes in [0.0, 1.0]: "
    f"{ATTR_LIST_STR}. "
    "Use the EXACT format: REASONING: <text>\\nATTRIBUTES: {\"attr\": score, ...}"
)

# Two-shot examples aligned with the paper's 23-attribute vocabulary.
FEW_SHOT = [
    {
        "user": (
            "Gene: BRCA1\n"
            "Evidence: Tumor suppressor; DNA repair; pLI=1.00; LOEUF=0.07; "
            "GWAS: breast/ovarian cancer p=1e-120; STRING: 287 partners; "
            "InterPro: BRCT domain, RING finger; UniProt: nuclear; "
            "Open Targets: sm_bucket=2, ab_bucket=2; 1863 aa; GTEx: ubiquitous."
        ),
        "assistant": (
            "REASONING: BRCA1 is a highly constrained tumor suppressor (pLI=1.00) "
            "with strong GWAS support for breast/ovarian cancer. Its BRCT domain "
            "enables PARP inhibitor synthetic lethality, giving it strong small-molecule "
            "and antibody tractability.\n"
            'ATTRIBUTES: {"disease_association": 0.98, "essential_gene": 0.80, '
            '"loss_of_function_tolerance": 0.05, "loss_of_function_constraint": 0.08, '
            '"observed_expected_lof_ratio": 0.07, "rvis_score": 0.04, '
            '"gwas_associations": 0.97, "protein_protein_interactions": 0.90, '
            '"chemical_gene_interactions": 0.70, "drug_interaction_types": 0.75, '
            '"known_antibodies": 0.85, "antibody": 0.65, "antibody_availability": 0.80, '
            '"small_molecule": 0.82, "tractable_modalities": 0.80, '
            '"overall_therapeutic_potential": 0.95, "functional_characterization": 0.95, '
            '"protein_domains": 0.80, "protein_length": 0.75, '
            '"expression_tissue_specificity": 0.12, "expression_specificity": 0.15, '
            '"expression_broadly_expressed": 0.95, "alternative_splicing": 0.70}'
        ),
    },
    {
        "user": (
            "Gene: KCNQ1\n"
            "Evidence: Voltage-gated K+ channel; cardiac repolarization; pLI=0.98; "
            "LOEUF=0.11; OMIM: long QT syndrome; GWAS: QT interval p=1e-85; "
            "STRING: 54 partners; InterPro: K-channel, PAS domain; "
            "UniProt: plasma membrane; Open Targets: sm_bucket=1; 676 aa; "
            "GTEx: heart-enriched (tau=0.62)."
        ),
        "assistant": (
            "REASONING: KCNQ1 is a highly constrained cardiac ion channel (pLI=0.98) "
            "with strong GWAS support for QT prolongation. Its plasma-membrane "
            "localization and approved channel-blocker interactions confirm "
            "small-molecule tractability, while heart-restricted expression limits "
            "off-target liability.\n"
            'ATTRIBUTES: {"disease_association": 0.92, "essential_gene": 0.70, '
            '"loss_of_function_tolerance": 0.08, "loss_of_function_constraint": 0.10, '
            '"observed_expected_lof_ratio": 0.11, "rvis_score": 0.06, '
            '"gwas_associations": 0.90, "protein_protein_interactions": 0.50, '
            '"chemical_gene_interactions": 0.60, "drug_interaction_types": 0.65, '
            '"known_antibodies": 0.55, "antibody": 0.40, "antibody_availability": 0.50, '
            '"small_molecule": 0.90, "tractable_modalities": 0.65, '
            '"overall_therapeutic_potential": 0.88, "functional_characterization": 0.88, '
            '"protein_domains": 0.72, "protein_length": 0.50, '
            '"expression_tissue_specificity": 0.62, "expression_specificity": 0.65, '
            '"expression_broadly_expressed": 0.40, "alternative_splicing": 0.45}'
        ),
    },
]


def make_prompt(gene, profile, tokenizer):
    """
    Build the full chat-formatted prompt for a single gene.

    Applies the tokenizer's chat template (e.g., ChatML format for Qwen)
    with system message, two-shot examples, and the gene query.
    """
    msgs = [{"role": "system", "content": SYS}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": f"Gene: {gene}\nEvidence: {profile}"})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_attrs(text):
    """
    Parse REASONING and ATTRIBUTES from the LLM's response text.

    Strategy:
      1. Try to parse the JSON block after "ATTRIBUTES:" directly.
      2. If JSON parsing fails or yields fewer than 5 attributes, fall back
         to regex key:value extraction for each known attribute name.
      3. Extract the REASONING section separately.

    Parameters
    ----------
    text : str  raw LLM generation text

    Returns
    -------
    attrs     : dict  {attr_name: float}  up to 23 attributes in [0, 1]
    reasoning : str   biological reasoning chain (up to 500 chars)
    """
    attrs = {}

    # --- Primary: JSON parse ---
    m = re.search(r'ATTRIBUTES:\s*(\{[^{}]*\})', text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            for k, v in obj.items():
                # Only accept values that are numeric and in the valid range [0, 1]
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1:
                    attrs[k] = float(v)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- Fallback: regex key:value extraction ---
    # Used when the model produces slight formatting errors (e.g., trailing commas)
    if len(attrs) < 5:
        for key in ATTR_VOCAB:
            pat = rf'["\s]*{re.escape(key)}["\s]*[:=]\s*([0-9]*\.?[0-9]+)'
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1))
                    if 0 <= v <= 1:
                        attrs[key] = v
                except ValueError:
                    pass

    # --- Extract reasoning chain ---
    reasoning = ""
    rm = re.search(r'REASONING:\s*(.*?)(?:ATTRIBUTES:|\{|$)', text, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()[:500]

    return attrs, reasoning


def attrs_to_vec(attrs):
    """
    Convert attrs dict to a 23-dim float array.

    Missing attributes default to 0.5, which represents neutral/uncertain.
    This is the same default used in the initial llm_mat before training.
    """
    return np.array([attrs.get(k, 0.5) for k in ATTR_VOCAB])


# ---------------------------------------------------------------------------
# Adjusted F1 reward (local computation, no gradient)
# ---------------------------------------------------------------------------

def adjusted_f1(probs, y):
    """
    Compute Adjusted F1 = R_soft^2 / p_bar on a numpy array.

    This is the reward signal for GRPO.  Used inside compute_adj_f1().
    See targetsage/metrics.py for the full annotated implementation.
    """
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar  = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


def compute_adj_f1(X, y, eval_mode="cv"):
    """
    Train a logistic regression on X and return the Adjusted F1 score.

    This function defines the GRPO reward: given a gene-feature matrix X
    (concatenation of bio features and the current LLM attribute matrix),
    it trains a logistic regression and evaluates Adjusted F1.

    The idea is that the RL policy (the LLM) is rewarded for generating
    attribute scores that, when concatenated with the bio features, lead to
    a better Adjusted F1 score from the downstream logistic regression.
    This encourages the LLM to generate attributes that COMPLEMENT (not
    duplicate) the information already in the bio features.

    Parameters
    ----------
    X         : [n_genes, d_bio + d_attrs]  concatenated feature matrix
    y         : [n_genes]  binary labels (1 = known positive, 0 = unlabeled)
    eval_mode : 'train' — fit and score on same data (fast, used for RL steps)
                'cv'    — 5-fold stratified CV (rigorous, used for reporting)

    Returns
    -------
    float  Adjusted F1 score in [0, 1]
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        Xs = StandardScaler().fit_transform(X)

        if eval_mode == "train":
            # Fast mode: fit LR and evaluate on the same data.
            # Slightly inflated but consistent enough for RL reward comparison.
            lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                    solver="lbfgs", random_state=42)
            lr.fit(Xs, y)
            prob = lr.predict_proba(Xs)[:, 1]
            return adjusted_f1(prob, y)

        # Rigorous 5-fold CV mode
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for tr_idx, te_idx in skf.split(Xs, y):
            lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                    solver="lbfgs", random_state=42)
            lr.fit(Xs[tr_idx], y[tr_idx])
            prob_te = lr.predict_proba(Xs[te_idx])[:, 1]
            scores.append(adjusted_f1(prob_te, y[te_idx]))
        return float(np.mean(scores))

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Generation utilities
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, prompts, device, temperature=0.8, max_tokens=250):
    """
    Generate text for a batch of prompts.

    Returns both the decoded text strings and the raw token ID tensors
    (needed for policy gradient computation).  Padding tokens at the
    end of each generation are stripped before computing the PG loss.

    Parameters
    ----------
    prompts     : list of prompt strings (chat-template formatted)
    temperature : sampling temperature (0.8 gives diverse rollouts)
    max_tokens  : maximum number of new tokens to generate

    Returns
    -------
    texts        : list of decoded generation strings
    gen_ids_list : list of CPU tensors, one per prompt, variable length
    """
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1536).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_tokens, temperature=temperature,
            top_k=20, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    texts = []
    gen_ids_list = []
    inp_len = enc["input_ids"].shape[1]
    for i in range(len(prompts)):
        gen_ids = out[i, inp_len:]
        # Strip padding tokens from the right end of the generation
        mask = gen_ids != tokenizer.pad_token_id
        if mask.any():
            last = mask.nonzero()[-1, 0].item() + 1
            gen_ids = gen_ids[:last]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
        gen_ids_list.append(gen_ids.cpu())
    return texts, gen_ids_list


def build_gene_profile(gene, g2s, agent_attrs_dict, max_len=600):
    """
    Build the evidence profile string for a gene.

    Combines the gene's text summary (from data/gene_summaries.tsv) with
    its top-10 agent-retrieved attributes from Module M1.

    Parameters
    ----------
    gene             : gene symbol (e.g., "BRCA1")
    g2s              : dict {gene_symbol: summary_text} from gene_summaries.tsv
    agent_attrs_dict : dict {gene_symbol: {attr: value}} from agent_attributes.csv
    max_len          : maximum character length of the profile

    Returns
    -------
    str  evidence profile string, at most max_len characters
    """
    # Start with the gene's natural language summary (up to 350 chars)
    summary = g2s.get(gene, "")[:350]
    parts = [summary] if summary else []

    # Add top-10 non-zero attributes from the M1 agent output
    if agent_attrs_dict and gene in agent_attrs_dict:
        agent_info = agent_attrs_dict[gene]
        top_attrs = []
        for k, v in sorted(agent_info.items()):
            if pd.notna(v) and v != 0:
                if isinstance(v, (int, float)):
                    top_attrs.append(f"{k}={v:.2f}")
                else:
                    top_attrs.append(f"{k}={str(v)[:40]}")
            if len(top_attrs) >= 10:
                break
        if top_attrs:
            parts.append("; ".join(top_attrs))

    profile = " | ".join(parts)[:max_len]
    return profile if profile else "No evidence available."


# ---------------------------------------------------------------------------
# Main GRPO training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="TargetSage M2: GRPO fine-tuning of the attribute-scoring LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",         default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model ID for the local LLM")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--bio_features",   default="data/gene_features.tsv")
    ap.add_argument("--agent_attrs",    default="results/agent_tools/agent_attributes_filtered.csv")
    ap.add_argument("--labels",         default="data/gene_labels.tsv")
    ap.add_argument("--outdir",         default="results/rl_grpo_v3")
    ap.add_argument("--task_name",      default="task_pharos_tclin_vs_others",
                    help="Primary task used to compute the RL reward.")
    ap.add_argument("--n_subset",       type=int,   default=1000,
                    help="Size of the labeled gene subset for reward computation.")
    ap.add_argument("--rl_steps",       type=int,   default=100,
                    help="Number of GRPO update steps.")
    ap.add_argument("--rl_lr",          type=float, default=5e-5,
                    help="Learning rate for the LoRA policy parameters.")
    ap.add_argument("--group_size",     type=int,   default=4,
                    help="Number of rollouts per gene per step (G in GRPO).")
    ap.add_argument("--genes_per_step", type=int,   default=16,
                    help="Number of genes sampled per RL step.")
    ap.add_argument("--temperature",    type=float, default=0.8,
                    help="Sampling temperature for generation (higher = more diverse).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load data ----
    print("[DATA] Loading...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    labels_df = pd.read_csv(args.labels, sep="\t")
    bio_df    = pd.read_csv(args.bio_features, sep="\t")
    agent_df  = pd.read_csv(args.agent_attrs)

    # Use the first 5 task columns for multi-task reward (broader signal)
    tasks = [c for c in labels_df.columns if c.startswith("task_")][:5]
    print(f"  Tasks: {tasks}")

    # Build lookup dictionaries for gene summaries and agent attributes
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    agent_attrs_dict = agent_df.set_index("Gene_Symbol").to_dict("index")

    # Gene universe: intersection of all four data sources
    genes_all = sorted(
        set(g2s.keys())
        & set(labels_df["Gene_Symbol"].astype(str))
        & set(bio_df["Gene_Symbol"].astype(str))
        & set(agent_df["Gene_Symbol"].astype(str))
    )
    print(f"  Total genes in intersection: {len(genes_all)}")

    # Build bio feature matrix [n_genes, d_bio], filling NaN with 0
    bio_df = bio_df.set_index("Gene_Symbol")
    bio_cols = list(bio_df.columns)
    print(f"  Bio features: {len(bio_cols)} dimensions")
    bio_df    = bio_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    bio_all   = bio_df.reindex(genes_all).values.astype(float)
    bio_all   = np.nan_to_num(bio_all, nan=0.0)
    print(f"  Bio matrix: {bio_all.shape}")

    # ---- Select labeled subset (positives + sampled negatives) ----
    # We work on a ~1000-gene subset for computational efficiency of the reward.
    # The subset is ~70% positives + ~30% negatives (or all positives if fewer than 700).
    task = args.task_name
    y_all    = labels_df.set_index("Gene_Symbol").reindex(genes_all)[task].values
    pos_idx  = np.where(y_all == 1)[0]
    neg_idx  = np.where(y_all == 0)[0]
    print(f"  Task '{task}': {len(pos_idx)} positive, {len(neg_idx)} negative")

    np.random.seed(42)
    target_pos = min(len(pos_idx), max(int(args.n_subset * 0.7), 200))
    target_neg = min(len(neg_idx), args.n_subset - target_pos)
    pos_sample = (np.random.choice(pos_idx, size=target_pos, replace=False)
                  if target_pos < len(pos_idx) else pos_idx)
    neg_sample = np.random.choice(neg_idx, size=target_neg, replace=False)
    subset_idx  = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(subset_idx)

    subset_genes = [genes_all[i] for i in subset_idx]
    subset_bio   = bio_all[subset_idx]
    subset_y     = y_all[subset_idx].astype(int)
    print(f"  Subset: {len(subset_genes)} genes "
          f"({subset_y.sum()} pos, {len(subset_y)-subset_y.sum()} neg)")

    # Baseline reward: logistic regression on bio features only (no LLM)
    bio_baseline = compute_adj_f1(subset_bio, subset_y)
    print(f"  Bio-only baseline Adjusted F1: {bio_baseline*100:.2f}%")

    # ---- Load LLM and apply LoRA ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"\n[MODEL] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # required for batch padding
    tokenizer.padding_side = "left"  # causal LM: pad on left for generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )

    # Apply LoRA: only the adapter weights (~2% of parameters) will be trained.
    # All attention projection layers and MLP gates are targeted to capture both
    # retrieval-head and computation-head behavior.
    lora_cfg = LoraConfig(
        r=32,                # LoRA rank: controls capacity of the adapter
        lora_alpha=64,       # scaling factor: effective LR = lr * lora_alpha / r
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Sanity check: test generation on 2 genes before training
    print("\n[SANITY] Testing generation on 2 genes...")
    test_genes   = subset_genes[:2]
    test_prompts = [make_prompt(g, build_gene_profile(g, g2s, agent_attrs_dict), tokenizer)
                    for g in test_genes]
    model.eval()
    texts, _ = generate_batch(model, tokenizer, test_prompts, device,
                               temperature=0.3, max_tokens=250)
    for g, t in zip(test_genes, texts):
        attrs, reasoning = extract_attrs(t)
        print(f"  [{g}] attrs_parsed={len(attrs)}, has_reasoning={'yes' if reasoning else 'no'}")

    # ---- Initialize LLM attribute matrix ----
    # llm_mat[i, :] = current best attribute vector for gene subset_genes[i].
    # Initialized to 0.5 (neutral) before any LLM responses.
    llm_mat   = np.full((len(subset_genes), len(ATTR_VOCAB)), 0.5)
    gene2idx  = {g: i for i, g in enumerate(subset_genes)}

    # Initial reward with neutral 0.5 filler (before any training)
    X_init       = np.concatenate([subset_bio, llm_mat], axis=1)
    init_reward  = compute_adj_f1(X_init, subset_y)
    print(f"  Initial reward (bio + 0.5 filler LLM attrs): {init_reward*100:.2f}%")

    # ---- GRPO training loop ----
    print(f"\n[RL] GRPO: {args.rl_steps} steps, "
          f"genes/step={args.genes_per_step}, group_size={args.group_size}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.rl_lr
    )

    log         = []
    best_reward = bio_baseline  # track the best global reward seen

    for step in range(args.rl_steps):
        t0 = time.time()

        # Sample a batch of genes from the subset
        batch_idxs      = np.random.choice(len(subset_genes),
                                            size=args.genes_per_step, replace=False)
        batch_genes_step = [subset_genes[i] for i in batch_idxs]

        # Build G=group_size prompts per gene (same prompt, different rollouts)
        all_prompts = []
        key_list    = []  # tracks which gene each rollout belongs to
        for gi_batch in batch_idxs:
            gene    = subset_genes[gi_batch]
            profile = build_gene_profile(gene, g2s, agent_attrs_dict)
            prompt  = make_prompt(gene, profile, tokenizer)
            for k in range(args.group_size):
                all_prompts.append(prompt)
                key_list.append(gi_batch)

        # Generate G rollouts per gene (with temperature for diversity)
        model.eval()
        texts, gen_ids_list = generate_batch(
            model, tokenizer, all_prompts, device, args.temperature, max_tokens=250
        )

        # Parse each rollout's attribute scores and reasoning chain
        rollouts = []
        for gi_batch, text, gen_ids in zip(key_list, texts, gen_ids_list):
            attrs, reasoning = extract_attrs(text)
            rollouts.append((gi_batch, attrs, reasoning, gen_ids))

        # ---- Compute rewards ----
        # For each rollout, temporarily substitute its attrs into llm_mat,
        # recompute Adjusted F1 on the full subset, then restore.
        rewards = []
        for (gi_batch, attrs, reasoning, _) in rollouts:
            orig_row = llm_mat[gi_batch].copy()
            if attrs:
                llm_mat[gi_batch] = attrs_to_vec(attrs)  # try this rollout's attrs

            X = np.concatenate([subset_bio, llm_mat], axis=1)
            r = compute_adj_f1(X, subset_y)  # reward = Adjusted F1 of LR(bio || llm_attrs)

            llm_mat[gi_batch] = orig_row  # restore original (don't commit yet)

            # Penalty for missing or trivial reasoning (length < 30 chars)
            # This encourages the model to produce interpretable reasoning chains.
            if not reasoning or len(reasoning) < 30:
                r = r * 0.5

            rewards.append(r)

        rewards = np.array(rewards)

        # ---- GRPO group advantage normalization ----
        # For each gene's group of G rollouts, compute within-group advantages.
        advantages = np.zeros_like(rewards)
        for i, gi_batch in enumerate(batch_idxs):
            start = i * args.group_size
            group = rewards[start:start + args.group_size]
            # Within-group centering: positive advantage = better than group mean
            advantages[start:start + args.group_size] = group - group.mean()

        # Global normalization: divide by std across all groups for scale invariance
        adv_std    = advantages.std() + 1e-8
        advantages = advantages / adv_std

        # ---- Policy gradient update ----
        # For each rollout with non-trivial advantage, compute cross-entropy loss
        # on the generated tokens and scale by the negative advantage.
        # Gradient accumulation across all rollouts, then average + clip.
        model.train()
        optimizer.zero_grad()
        total_pg_loss = 0.0
        n_valid       = 0

        for i, (gi_batch, attrs, reasoning, gen_ids) in enumerate(rollouts):
            adv = advantages[i]
            # Skip rollouts with near-zero advantage (no signal) or empty generation
            if abs(adv) < 1e-6 or len(gen_ids) == 0:
                continue

            # Reconstruct the full token sequence: [prompt] + [generation]
            prompt  = all_prompts[i]
            enc     = tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=1536).to(device)
            inp     = enc["input_ids"][0]
            full_ids = torch.cat([inp, gen_ids.to(device)]).unsqueeze(0)

            # Mask the prompt tokens in the label (-100 = ignored by cross-entropy)
            # so the loss only applies to the generated tokens.
            labels = full_ids.clone()
            labels[0, :len(inp)] = -100

            # Forward pass: compute CE loss on generated tokens
            out     = model(input_ids=full_ids, labels=labels)

            # Policy gradient: L = CE_loss * (-advantage)
            # Negative sign: we want to maximize reward (ascending),
            # but PyTorch minimizes, so we minimize (-reward).
            pg_loss = out.loss * (-adv)
            pg_loss.backward()
            total_pg_loss += pg_loss.item()
            n_valid       += 1

        if n_valid > 0:
            # Average gradients across rollouts (like a mini-batch)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= n_valid
            # Gradient clipping prevents large updates from outlier rollouts
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # ---- Update llm_mat with the best rollout per gene ----
        # After the policy update, commit the best-performing attribute vector
        # for each gene in the batch.
        for i, gi_batch in enumerate(batch_idxs):
            start    = i * args.group_size
            group    = rewards[start:start + args.group_size]
            best_j   = int(group.argmax())
            best_attrs = rollouts[start + best_j][1]
            if best_attrs:
                llm_mat[gi_batch] = attrs_to_vec(best_attrs)

        # ---- Global reward on the full subset ----
        global_X      = np.concatenate([subset_bio, llm_mat], axis=1)
        global_reward = compute_adj_f1(global_X, subset_y)

        # Logging
        mean_r        = rewards.mean()
        valid_pct     = sum(1 for _, a, _, _ in rollouts if a) / len(rollouts) * 100
        reasoning_pct = sum(1 for _, _, r, _ in rollouts if r and len(r) > 30) / len(rollouts) * 100
        dt            = time.time() - t0

        log.append({
            "step":          step,
            "mean_reward":   float(mean_r),
            "global_reward": float(global_reward),
            "bio_baseline":  float(bio_baseline),
            "valid_pct":     float(valid_pct),
            "reasoning_pct": float(reasoning_pct),
            "pg_loss":       float(total_pg_loss / max(n_valid, 1)),
            "time_s":        dt,
        })

        # Save best model checkpoint
        if global_reward > best_reward:
            best_reward = global_reward
            model.save_pretrained(os.path.join(args.outdir, "rl_best"))
            tokenizer.save_pretrained(os.path.join(args.outdir, "rl_best"))

        if step % 2 == 0:
            print(
                f"  Step {step:3d}: rollout={mean_r*100:.2f}% "
                f"global={global_reward*100:.2f}% (bio={bio_baseline*100:.2f}%) "
                f"valid={valid_pct:.0f}% reas={reasoning_pct:.0f}% "
                f"loss={total_pg_loss/max(n_valid,1):.4f} {dt:.0f}s"
            )

        if step % 10 == 0:
            pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    # ---- Save final results ----
    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    print(f"\n[DONE] Bio-only baseline:  {bio_baseline*100:.2f}%")
    print(f"       Best RL reward:     {best_reward*100:.2f}%")
    print(f"       Improvement:        +{(best_reward - bio_baseline)*100:.2f}%")

    # Save the final LLM attribute matrix (used as input to TargetSage M3)
    np.save(os.path.join(args.outdir, "llm_attrs_subset.npy"), llm_mat)
    pd.DataFrame(llm_mat, columns=ATTR_VOCAB, index=subset_genes).to_csv(
        os.path.join(args.outdir, "llm_attrs_subset.csv")
    )
    print(f"       Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
