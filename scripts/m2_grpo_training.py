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

GRPO Reward Function (Self-Supervised Dynamic Embeddings)
----------------------------------------------------------
At each RL step:
  1. Sample a batch of G=4 rollouts per gene.
  2. For each rollout, substitute the generated attribute vector (scaled via
     sc_attr) into a copy of the scaled attr matrix Xa_mod.
  3. With --self_embed (default): extract mean-pooled last-layer hidden states
     of the GENERATED tokens from the reasoning policy, project via fixed W_proj
     to 256-dim, and substitute into Xe_mod[gi_batch].  This creates a
     self-supervised feedback loop: better reasoning → richer LLM hidden states
     → higher Xe representation quality → higher M3 reward → stronger GRPO signal.
  4. Pass (Xa_mod, Xe_mod) through each frozen proxy M3 discriminator
     (pre-trained once on base LLM embeddings before GRPO) via a forward pass.
  5. Reward r = macro-average Adjusted F1 = mean_t(R_soft_t^2 / p̄_t)
     across all T=15 tasks.
  6. A penalty factor 0.5x is applied if the reasoning chain is absent or
     shorter than 30 characters.

Key design properties:
  - M3 proxy models are pre-trained on LLM hidden-state embeddings from the
    base (pre-GRPO) model so their embedding input distribution is consistent
    with what they receive during reward evaluation.
  - W_proj is fixed throughout training: M3 sees a stable emb → latent mapping.
  - No external API calls are needed for embeddings during GRPO training.

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
import torch.nn.functional as F

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from targetsage import TargetSage, nnpu_loss, TASKS, TASK_DISPLAY, load_prior_map


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
# Adjusted F1 (metric helper)
# ---------------------------------------------------------------------------

def adjusted_f1(probs, y):
    """Compute Adjusted F1 = R_soft^2 / p_bar.  See targetsage/metrics.py."""
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar  = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


# ---------------------------------------------------------------------------
# Dynamic implicit embeddings — LLM hidden-state extraction
# ---------------------------------------------------------------------------

def extract_lm_emb_batch(model, tokenizer, texts, device, W_proj,
                          max_len=512, batch_size=8):
    """
    Extract mean-pooled last-layer hidden states for a list of text strings and
    project them to W_proj.shape[1] dimensions via a fixed random projection.

    This provides a self-supervised implicit embedding of each text that lives
    in the same space as the M3 reward model's embedding input, enabling
    per-rollout trace embeddings without any external API calls.

    Parameters
    ----------
    texts     : list of str  (gene evidence profiles or reasoning traces)
    W_proj    : [d_llm, d_emb_proj]  fixed random projection matrix
    max_len   : maximum token length (longer texts are truncated)
    batch_size: number of texts processed per forward pass

    Returns
    -------
    np.ndarray  [n, d_emb_proj]  projected embeddings, one row per text
    """
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_len,
            ).to(device)
            out    = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()          # [bs, seq, d_llm]
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # [bs, d_llm]
            all_embs.append(pooled.cpu().numpy() @ W_proj)  # [bs, d_emb_proj]
    return np.vstack(all_embs)  # [n, d_emb_proj]


# ---------------------------------------------------------------------------
# M3 pre-training and reward computation
# ---------------------------------------------------------------------------

def preprocess_subset(Xb, Xa, Xe, emb_pca_dim=256, seed=42):
    """
    Impute missing values, optionally compress embeddings with PCA,
    and StandardScale all three modalities.  Returns scaled arrays plus
    sc_attr and sc_emb so new rollout vectors can be scaled consistently.
    """
    imp_b = SimpleImputer(strategy="median")
    imp_a = SimpleImputer(strategy="median")
    imp_e = SimpleImputer(strategy="median")
    Xb = imp_b.fit_transform(Xb)
    Xa = imp_a.fit_transform(Xa)
    Xe = imp_e.fit_transform(Xe)

    if emb_pca_dim > 0 and Xe.shape[1] > emb_pca_dim:
        pca = PCA(n_components=min(emb_pca_dim, Xe.shape[1]), random_state=seed)
        Xe = pca.fit_transform(Xe)

    sc_b = StandardScaler(); sc_a = StandardScaler(); sc_e = StandardScaler()
    Xb = sc_b.fit_transform(Xb)
    Xa = sc_a.fit_transform(Xa)
    Xe = sc_e.fit_transform(Xe)

    return Xb, Xa, Xe, sc_a, sc_e


def train_m3_for_reward(Xb, Xa, Xe, y, pi,
                         device, d_latent=256, head_h=512, dropout=0.2,
                         warmup_epochs=10, nnpu_epochs=25,
                         batch_size=512, lr=2e-4, beta=0.6):
    """
    Train a TargetSage (Module 3) model on the reward subset using
    preprocessed bio/attr/emb arrays and the hybrid class prior pi.

    The trained model is returned in eval mode with all gradients disabled —
    it serves as a frozen reward proxy during GRPO training.

    Parameters
    ----------
    Xb, Xa, Xe : [n, d_*]  scaled feature arrays for the subset
    y           : [n]       binary labels (1 = positive, 0 = unlabeled)
    pi          : float     hybrid class prior P(Y=1) for this task
    """
    model = TargetSage(
        d_bio=Xb.shape[1], d_attr=Xa.shape[1], d_emb=Xe.shape[1],
        d_latent=d_latent, head_h=head_h, dropout=dropout, fusion="gated",
    ).to(device)

    idx_p = np.where(y == 1)[0]
    idx_u = np.where(y == 0)[0]
    if len(idx_p) < 10 or len(idx_u) < 10:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    # --- Stage 1: BCE warmup ---
    n_pos, n_unl = len(idx_p), len(idx_u)
    bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_unl / max(n_pos, 1)], device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(warmup_epochs):
        perm = np.random.permutation(len(y))   # re-shuffle every epoch
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            lp, _, _ = model.forward_logits(
                torch.from_numpy(Xb[idx]).float().to(device),
                torch.from_numpy(Xa[idx]).float().to(device),
                torch.from_numpy(Xe[idx]).float().to(device),
            )
            opt.zero_grad()
            bce(lp, torch.from_numpy(y[idx].astype(np.float32)).to(device)).backward()
            opt.step()

    # --- Compute semantic weights once (before nnPU loop) ---
    model.eval()
    with torch.no_grad():
        lp_all, _, h_e = model.forward_logits(
            torch.from_numpy(Xb).float().to(device),
            torch.from_numpy(Xa).float().to(device),
            torch.from_numpy(Xe).float().to(device),
        )
        prob_all = torch.sigmoid(lp_all)
        centroid = F.normalize(h_e[idx_p].mean(0, keepdim=True), dim=1)
        h_u_norm = F.normalize(h_e[idx_u], dim=1)
        sim_u    = ((h_u_norm * centroid).sum(1) + 1.0) / 2.0
        w_u = torch.clamp(
            beta * prob_all[idx_u] + (1 - beta) * sim_u, 0, 1
        ).cpu().numpy()

    # --- Stage 2: nnPU training ---
    u_map = {int(idx_u[i]): i for i in range(len(idx_u))}
    n_batch = max(1, min(len(idx_p), len(idx_u)) // batch_size)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(nnpu_epochs):
        for _ in range(n_batch):
            bp = np.random.choice(idx_p, batch_size, replace=True)
            bu = np.random.choice(idx_u, batch_size, replace=True)
            lp_p, _, _ = model.forward_logits(
                torch.from_numpy(Xb[bp]).float().to(device),
                torch.from_numpy(Xa[bp]).float().to(device),
                torch.from_numpy(Xe[bp]).float().to(device),
            )
            lp_u, _, _ = model.forward_logits(
                torch.from_numpy(Xb[bu]).float().to(device),
                torch.from_numpy(Xa[bu]).float().to(device),
                torch.from_numpy(Xe[bu]).float().to(device),
            )
            w_batch = torch.from_numpy(
                w_u[[u_map[int(g)] for g in bu]]
            ).float().to(device)
            loss = nnpu_loss(lp_p, lp_u, pi=pi, w_u=w_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Freeze: no gradients flow through M3 during GRPO training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def compute_m3_reward(m3_models, Xb, Xa_modified, Xe, y_dict, device):
    """
    Compute the macro-averaged Adjusted F1 reward across all tasks by running
    the modified attribute matrix through each frozen M3 discriminator.

    Parameters
    ----------
    m3_models   : dict {task_key: frozen TargetSage model}
    Xb          : [n, d_bio]  scaled bio features (fixed throughout GRPO)
    Xa_modified : [n, d_attr] scaled attr matrix with one gene's row substituted
    Xe          : [n, d_emb]  scaled embeddings (fixed throughout GRPO)
    y_dict      : dict {task_key: [n] binary labels}

    Returns
    -------
    float  macro-average Adjusted F1 across all tasks
    """
    Xb_t = torch.from_numpy(Xb).float().to(device)
    Xa_t = torch.from_numpy(Xa_modified).float().to(device)
    Xe_t = torch.from_numpy(Xe).float().to(device)

    scores = []
    for task, m3 in m3_models.items():
        y = y_dict[task]
        if y.sum() == 0:
            continue
        lp, _, _ = m3.forward_logits(Xb_t, Xa_t, Xe_t)
        probs = torch.sigmoid(lp).cpu().numpy()
        scores.append(adjusted_f1(probs, y))

    return float(np.mean(scores)) if scores else 0.0


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
    ap.add_argument("--scores",         default="data/features_llm_structured_scores.csv",
                    help="Pre-GRPO LLM attribute scores (used to pre-train M3 reward models).")
    ap.add_argument("--embeddings",     default="data/features_llm_embedding.csv",
                    help="LLM embedding CSV (required for M3 reward).")
    ap.add_argument("--prior_csv",      default="data/prior_task_summary.csv",
                    help="LLM-derived class prior per task (required for M3 nnPU).")
    ap.add_argument("--agent_attrs",    default="results/agent_tools/agent_attributes_filtered.csv")
    ap.add_argument("--labels",         default="data/gene_labels.tsv")
    ap.add_argument("--outdir",         default="results/rl_grpo_v3")
    ap.add_argument("--n_subset",       type=int,   default=1000,
                    help="Size of the labeled gene subset for reward computation.")
    ap.add_argument("--emb_pca_dim",    type=int,   default=256,
                    help="PCA dimension for LLM embeddings before M3.")
    ap.add_argument("--alpha",          type=float, default=0.6,
                    help="Hybrid prior mixing coefficient (pi = alpha*pi_data + (1-alpha)*pi_llm).")
    ap.add_argument("--beta",           type=float, default=0.6,
                    help="Semantic guidance weight for nnPU soft confidence.")
    ap.add_argument("--pi_cap",         type=float, default=0.10,
                    help="Maximum allowed class prior.")
    ap.add_argument("--m3_warmup",      type=int,   default=10,
                    help="BCE warmup epochs for M3 pre-training.")
    ap.add_argument("--m3_nnpu",        type=int,   default=25,
                    help="nnPU training epochs for M3 pre-training.")
    ap.add_argument("--self_embed",     action="store_true", default=True,
                    help="Use LLM last-layer hidden states as implicit embeddings for "
                         "M3 pre-training and per-rollout GRPO reward (paper-faithful). "
                         "Pass --no_self_embed to fall back to static OpenAI embeddings.")
    ap.add_argument("--no_self_embed",  dest="self_embed", action="store_false")
    ap.add_argument("--lm_proj_seed",   type=int,   default=42,
                    help="RNG seed for the fixed random W_proj projection matrix.")
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
    scores_df = pd.read_csv(args.scores)          # pre-GRPO LLM attr scores (for M3 pre-train)
    emb_df    = pd.read_csv(args.embeddings)       # LLM embeddings (for M3)
    agent_df  = pd.read_csv(args.agent_attrs)
    prior_map = load_prior_map(args.prior_csv)     # {display_name: pi_llm}

    # All 15 tasks used for multi-task M3 reward (macro-avg Adjusted F1)
    tasks = [c for c in labels_df.columns if c.startswith("task_")]
    print(f"  Tasks ({len(tasks)}): {tasks}")

    # Build lookup dictionaries for gene summaries and agent attributes
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    agent_attrs_dict = agent_df.set_index("Gene_Symbol").to_dict("index")

    # Gene universe: intersection of all data sources
    genes_all = sorted(
        set(g2s.keys())
        & set(labels_df["Gene_Symbol"].astype(str))
        & set(bio_df["Gene_Symbol"].astype(str))
        & set(scores_df["Gene_Symbol"].astype(str))
        & set(emb_df["Gene_Symbol"].astype(str))
        & set(agent_df["Gene_Symbol"].astype(str))
    )
    print(f"  Total genes in intersection: {len(genes_all)}")

    # Build raw feature matrices aligned to genes_all
    bio_df    = bio_df.set_index("Gene_Symbol").apply(pd.to_numeric, errors="coerce")
    scores_df = scores_df.set_index("Gene_Symbol")
    emb_df    = emb_df.set_index("Gene_Symbol")

    attr_cols      = [c for c in scores_df.columns if c in ATTR_VOCAB]
    attr_col_mask  = np.array([ATTR_VOCAB.index(c) for c in attr_cols])
    emb_cols       = [c for c in emb_df.columns]

    bio_all  = bio_df.reindex(genes_all).values.astype(float)
    attr_all = scores_df[attr_cols].reindex(genes_all).values.astype(float)
    emb_all  = emb_df[emb_cols].reindex(genes_all).values.astype(float)
    bio_all  = np.nan_to_num(bio_all,  nan=0.0)
    attr_all = np.nan_to_num(attr_all, nan=0.5)
    emb_all  = np.nan_to_num(emb_all,  nan=0.0)
    print(f"  Shapes — bio:{bio_all.shape}  attr:{attr_all.shape}  emb:{emb_all.shape}")

    # ---- Select labeled subset balanced across tasks ----
    # Use the first task to define positives for subset sampling (Clinical Targets),
    # then include label columns for all 15 tasks in y_dict_all.
    labels_idx = labels_df.set_index("Gene_Symbol").reindex(genes_all)
    anchor_task = tasks[0]  # for stratified sampling
    y_anchor = labels_idx[anchor_task].fillna(0).values.astype(int)
    pos_idx  = np.where(y_anchor == 1)[0]
    neg_idx  = np.where(y_anchor == 0)[0]

    np.random.seed(42)
    target_pos = min(len(pos_idx), max(int(args.n_subset * 0.7), 200))
    target_neg = min(len(neg_idx), args.n_subset - target_pos)
    pos_sample = (np.random.choice(pos_idx, size=target_pos, replace=False)
                  if target_pos < len(pos_idx) else pos_idx)
    neg_sample = np.random.choice(neg_idx, size=target_neg, replace=False)
    subset_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(subset_idx)

    subset_genes = [genes_all[i] for i in subset_idx]
    subset_bio_raw  = bio_all[subset_idx]
    subset_attr_raw = attr_all[subset_idx]   # pre-GRPO scores (for M3 pre-train)
    subset_emb_raw  = emb_all[subset_idx]

    # y_dict: binary labels per task for the subset
    y_dict = {}
    for t in tasks:
        if t in labels_idx.columns:
            y_dict[t] = labels_idx[t].fillna(0).values[subset_idx].astype(int)

    subset_y = y_dict.get(anchor_task, np.zeros(len(subset_idx), dtype=int))
    print(f"  Subset: {len(subset_genes)} genes "
          f"({subset_y.sum()} pos in anchor task, {len(subset_y)-subset_y.sum()} neg)")

    # ---- Load LLM and apply LoRA ----
    # Loaded before M3 pre-training so that (when --self_embed) the LLM's last-layer
    # hidden states serve as the implicit embedding Xe for both M3 pre-training and
    # per-rollout GRPO reward, creating a fully self-contained dynamic embedding pipeline.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"\n[MODEL] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )

    # Apply LoRA: only the adapter weights (~2% of parameters) will be trained.
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
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

    # ---- Dynamic implicit embeddings: fixed random projection from LLM hidden states ----
    # W_proj maps LLM last-layer hidden states (dim d_llm) → emb_pca_dim via a fixed
    # Kaiming-scaled random projection, initialized once before any training.
    # With --self_embed (default):
    #   (a) M3 pre-training uses mean-pooled hidden states of gene evidence profiles as Xe,
    #       so M3 learns to interpret the LLM's internal representation of each gene.
    #   (b) GRPO reward substitutes Xe[gi_batch] with the mean-pooled hidden states of the
    #       generated reasoning trace, so both explicit attr scores AND reasoning depth
    #       jointly shape the reward signal — better reasoning → richer hidden states →
    #       higher M3 reward → stronger GRPO gradient.
    d_llm  = model.config.hidden_size
    W_proj = (np.random.default_rng(args.lm_proj_seed)
              .standard_normal((d_llm, args.emb_pca_dim))
              .astype(np.float32) / np.sqrt(d_llm))

    if args.self_embed:
        print(f"\n[EMBED] Extracting base LLM hidden-state embeddings for "
              f"{len(subset_genes)} genes (pre-GRPO model)...")
        subset_profiles = [
            build_gene_profile(g, g2s, agent_attrs_dict) for g in subset_genes
        ]
        emb_for_m3 = extract_lm_emb_batch(
            model, tokenizer, subset_profiles, device, W_proj,
            max_len=512, batch_size=8,
        )
        print(f"  LLM hidden-state emb shape: {emb_for_m3.shape}")
    else:
        emb_for_m3 = subset_emb_raw  # fall back to pre-computed OpenAI embeddings

    # ---- Preprocess subset (impute / PCA / scale) ----
    # sc_attr and sc_emb are saved so new rollout vectors can be scaled consistently.
    subset_bio, subset_attr_scaled, subset_emb_scaled, sc_attr, sc_emb = preprocess_subset(
        subset_bio_raw, subset_attr_raw, emb_for_m3,
        emb_pca_dim=args.emb_pca_dim, seed=42,
    )

    # ---- Pre-train one frozen M3 model per task ----
    print(f"\n[M3] Pre-training frozen M3 reward models for {len(tasks)} tasks...")
    m3_models = {}
    for t in tasks:
        y_t = y_dict.get(t, None)
        if y_t is None or y_t.sum() < 10:
            continue
        display   = TASK_DISPLAY.get(t, t)
        pi_llm    = prior_map.get(display, float(y_t.mean()))
        pi_data   = float(y_t.mean())
        pi_used   = float(np.clip(
            args.alpha * pi_data + (1 - args.alpha) * pi_llm,
            float(y_t.mean()), args.pi_cap,
        ))
        m3_models[t] = train_m3_for_reward(
            subset_bio, subset_attr_scaled, subset_emb_scaled, y_t, pi_used,
            device=device,
            warmup_epochs=args.m3_warmup, nnpu_epochs=args.m3_nnpu,
            batch_size=512, lr=2e-4, beta=args.beta,
        )
        print(f"  [{display}] pi_used={pi_used:.4f}  M3 ready (frozen)")

    print(f"  M3 models ready: {len(m3_models)} tasks")

    # ---- Initialize LLM attribute matrix ----
    # llm_mat[i, :] = current best RAW (unscaled) attribute vector for gene subset_genes[i].
    # Initialized to pre-GRPO LLM scores (subset_attr_raw) as the starting point.
    # Shape: (n_subset, len(attr_cols))
    llm_mat  = subset_attr_raw.copy()
    gene2idx = {g: i for i, g in enumerate(subset_genes)}

    # M3 baseline: reward using pre-GRPO (unoptimized) LLM attr scores + base embeddings
    m3_baseline = compute_m3_reward(
        m3_models, subset_bio, subset_attr_scaled, subset_emb_scaled, y_dict, device
    )
    print(f"  M3 baseline reward (pre-GRPO attr scores): {m3_baseline*100:.2f}%")

    # ---- GRPO training loop ----
    print(f"\n[RL] GRPO: {args.rl_steps} steps, "
          f"genes/step={args.genes_per_step}, group_size={args.group_size}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.rl_lr
    )

    log         = []
    best_reward = m3_baseline  # track the best global reward seen

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
        # For each rollout, build modified attr and (when --self_embed) emb matrices,
        # then pass through frozen M3 models to get macro-avg Adjusted F1 reward.
        rewards = []
        for rollout_idx, (gi_batch, attrs, reasoning, gen_ids) in enumerate(rollouts):
            Xa_mod = subset_attr_scaled.copy()
            Xe_mod = subset_emb_scaled.copy()

            if attrs:
                raw_row = attrs_to_vec(attrs)[attr_col_mask]
                Xa_mod[gi_batch] = sc_attr.transform(raw_row.reshape(1, -1))[0]

            if args.self_embed and len(gen_ids) > 0:
                # Substitute gene gi_batch's embedding row with the mean-pooled
                # last-layer hidden states of the generated reasoning trace.
                # This makes the reward sensitive to reasoning quality beyond just
                # the explicit attribute scores.
                prompt_i = all_prompts[rollout_idx]
                enc_i    = tokenizer(
                    prompt_i, return_tensors="pt", truncation=True, max_length=1536,
                ).to(device)
                inp_len  = enc_i["input_ids"].shape[1]
                full_seq = torch.cat(
                    [enc_i["input_ids"][0], gen_ids.to(device)]
                ).unsqueeze(0)
                with torch.no_grad():
                    out_h = model(input_ids=full_seq, output_hidden_states=True)
                    gen_h = out_h.hidden_states[-1][0, inp_len:, :].float()
                    if gen_h.shape[0] > 0:
                        trace_proj = gen_h.mean(0).cpu().numpy() @ W_proj
                        Xe_mod[gi_batch] = sc_emb.transform(
                            trace_proj.reshape(1, -1)
                        )[0]

            r = compute_m3_reward(m3_models, subset_bio, Xa_mod, Xe_mod, y_dict, device)

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
                llm_mat[gi_batch] = attrs_to_vec(best_attrs)[attr_col_mask]

        # ---- Global reward on the full subset ----
        # Uses base embeddings (not per-rollout trace embeddings) as a stable
        # reference signal tracking how the committed llm_mat has improved.
        global_Xa_scaled = sc_attr.transform(llm_mat)
        global_reward    = compute_m3_reward(
            m3_models, subset_bio, global_Xa_scaled, subset_emb_scaled, y_dict, device
        )

        # Logging
        mean_r        = rewards.mean()
        valid_pct     = sum(1 for _, a, _, _ in rollouts if a) / len(rollouts) * 100
        reasoning_pct = sum(1 for _, _, r, _ in rollouts if r and len(r) > 30) / len(rollouts) * 100
        dt            = time.time() - t0

        log.append({
            "step":          step,
            "mean_reward":   float(mean_r),
            "global_reward": float(global_reward),
            "m3_baseline":   float(m3_baseline),
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
                f"global={global_reward*100:.2f}% (baseline={m3_baseline*100:.2f}%) "
                f"valid={valid_pct:.0f}% reas={reasoning_pct:.0f}% "
                f"loss={total_pg_loss/max(n_valid,1):.4f} {dt:.0f}s"
            )

        if step % 10 == 0:
            pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    # ---- Save final results ----
    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    print(f"\n[DONE] M3 baseline (pre-GRPO): {m3_baseline*100:.2f}%")
    print(f"       Best RL reward:        {best_reward*100:.2f}%")
    print(f"       Improvement:           +{(best_reward - m3_baseline)*100:.2f}%")

    # Save the final LLM attribute matrix (used as input to TargetSage M3)
    np.save(os.path.join(args.outdir, "llm_attrs_subset.npy"), llm_mat)
    pd.DataFrame(llm_mat, columns=attr_cols, index=subset_genes).to_csv(
        os.path.join(args.outdir, "llm_attrs_subset.csv")
    )
    print(f"       Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
