#!/usr/bin/env python3
"""
GRPO v3: Direct RL on a labeled subset with bio+LLM fusion reward.

Core idea (matches advisor spec):
- Work on a labeled SUBSET (~1000 genes: positives + sampled negatives)
- Local model generates reasoning chain + 20-dim LLM attribute scores per gene
- Reward = Adj F1 of a classifier trained on [bio || llm_attrs] features
- The LLM must generate attrs that COMPLEMENT bio features to increase reward
- GRPO optimizes local model to produce the most informative attrs + reasoning

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u scripts/rl_grpo_v3.py
"""

import os, sys, json, re, argparse, time
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── Fixed 20-dim therapeutic attribute vocabulary ────────────────────────────
ATTR_VOCAB = [
    "disease_association", "druggability", "loss_of_function_tolerance",
    "safety_profile", "tissue_specificity", "cancer_relevance",
    "tractable_small_molecule", "tractable_antibody", "tractable_protac",
    "enzyme_activity", "membrane_localization", "secreted_protein",
    "pathway_centrality", "gwas_support", "essential_gene",
    "novelty", "expression_breadth", "protein_interactions",
    "biomarker_potential", "clinical_validation",
]

ATTR_LIST_STR = ", ".join(ATTR_VOCAB)

SYS = (
    "You are a drug discovery scientist evaluating therapeutic targets. "
    "Given a gene's evidence profile, write a 2-3 sentence biological reasoning chain, "
    f"then score the gene on these {len(ATTR_VOCAB)} attributes in [0.0, 1.0]: "
    f"{ATTR_LIST_STR}. "
    "Use the EXACT format: REASONING: <text>\\nATTRIBUTES: {\"attr\": score, ...}"
)

FEW_SHOT = [
    {
        "user": "Gene: BRCA1\nEvidence: Breast cancer susceptibility; DNA repair; pLI=1.00; GWAS: breast/ovarian cancer; tractable small molecule/antibody.",
        "assistant": (
            "REASONING: BRCA1 is a tumor suppressor with strong genetic constraint (pLI=1.00) and validated GWAS linkage to breast/ovarian cancer. Its DNA repair function makes it a clinically validated target for PARP inhibitor synthetic lethality.\n"
            'ATTRIBUTES: {"disease_association": 0.95, "druggability": 0.80, "loss_of_function_tolerance": 0.05, "safety_profile": 0.60, "tissue_specificity": 0.30, "cancer_relevance": 0.95, "tractable_small_molecule": 0.85, "tractable_antibody": 0.50, "tractable_protac": 0.40, "enzyme_activity": 0.70, "membrane_localization": 0.10, "secreted_protein": 0.05, "pathway_centrality": 0.85, "gwas_support": 0.95, "essential_gene": 0.60, "novelty": 0.20, "expression_breadth": 0.70, "protein_interactions": 0.80, "biomarker_potential": 0.85, "clinical_validation": 0.90}'
        ),
    },
    {
        "user": "Gene: ACLY\nEvidence: ATP-citrate lyase; lipid metabolism; pLI=0.92; cancer metabolism; bempedoic acid approved.",
        "assistant": (
            "REASONING: ACLY is a metabolic enzyme validated as a small molecule target through bempedoic acid. It links cytosolic acetyl-CoA to lipid biosynthesis and cancer metabolism, with moderate loss-of-function constraint indicating functional importance.\n"
            'ATTRIBUTES: {"disease_association": 0.75, "druggability": 0.95, "loss_of_function_tolerance": 0.08, "safety_profile": 0.75, "tissue_specificity": 0.40, "cancer_relevance": 0.70, "tractable_small_molecule": 0.98, "tractable_antibody": 0.10, "tractable_protac": 0.20, "enzyme_activity": 0.95, "membrane_localization": 0.05, "secreted_protein": 0.05, "pathway_centrality": 0.80, "gwas_support": 0.60, "essential_gene": 0.40, "novelty": 0.30, "expression_breadth": 0.85, "protein_interactions": 0.60, "biomarker_potential": 0.60, "clinical_validation": 0.90}'
        ),
    },
]


def make_prompt(gene, profile, tokenizer):
    msgs = [{"role": "system", "content": SYS}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": f"Gene: {gene}\nEvidence: {profile}"})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_attrs(text):
    """Extract REASONING and ATTRIBUTES from text."""
    attrs = {}
    # JSON
    m = re.search(r'ATTRIBUTES:\s*(\{[^{}]*\})', text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            for k, v in obj.items():
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1:
                    attrs[k] = float(v)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: match key:value patterns
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

    # Reasoning
    reasoning = ""
    rm = re.search(r'REASONING:\s*(.*?)(?:ATTRIBUTES:|\{|$)', text, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()[:500]
    return attrs, reasoning


def attrs_to_vec(attrs):
    """Convert attrs dict to a 20-dim vector (missing → 0.5)."""
    return np.array([attrs.get(k, 0.5) for k in ATTR_VOCAB])


# ── Reward ───────────────────────────────────────────────────────────────────
def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


def compute_adj_f1(X, y, eval_mode="cv"):
    """Fit LR on X, return Adj F1.

    eval_mode:
      'train'  - fit and evaluate on same data (fast, inflated, good for RL reward)
      'cv'     - 5-fold stratified CV, mean test Adj F1 (rigorous, slow)
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        Xs = StandardScaler().fit_transform(X)

        if eval_mode == "train":
            lr = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced",
                                    solver="lbfgs", random_state=42)
            lr.fit(Xs, y)
            prob = lr.predict_proba(Xs)[:, 1]
            return adjusted_f1(prob, y)

        # CV mode
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


# ── Generation ───────────────────────────────────────────────────────────────
def generate_batch(model, tokenizer, prompts, device, temperature=0.8, max_tokens=250):
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
        # Skip padding at end
        mask = gen_ids != tokenizer.pad_token_id
        if mask.any():
            last = mask.nonzero()[-1, 0].item() + 1
            gen_ids = gen_ids[:last]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
        gen_ids_list.append(gen_ids.cpu())
    return texts, gen_ids_list


def build_gene_profile(gene, g2s, agent_attrs_dict, max_len=600):
    summary = g2s.get(gene, "")[:350]
    parts = [summary] if summary else []

    if agent_attrs_dict and gene in agent_attrs_dict:
        agent_info = agent_attrs_dict[gene]
        top_attrs = []
        for k, v in sorted(agent_info.items()):
            if pd.notna(v) and v != 0:
                if isinstance(v, (int, float)):
                    top_attrs.append(f"{k}={v:.2f}")
                else:
                    s = str(v)[:40]
                    top_attrs.append(f"{k}={s}")
            if len(top_attrs) >= 10:
                break
        if top_attrs:
            parts.append("; ".join(top_attrs))

    profile = " | ".join(parts)[:max_len]
    return profile if profile else "No evidence available."


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--bio_features", default="data/gene_features.tsv")
    ap.add_argument("--agent_attrs", default="results/agent_tools/agent_attributes_filtered.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results/rl_grpo_v3")
    ap.add_argument("--task_name", default="task_pharos_tclin_vs_others",
                    help="Primary task to use for RL reward.")
    ap.add_argument("--n_subset", type=int, default=1000, help="Labeled subset size.")
    ap.add_argument("--rl_steps", type=int, default=100)
    ap.add_argument("--rl_lr", type=float, default=5e-5)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--genes_per_step", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ────────────────────────────────────────────────────────
    print("[DATA] Loading...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    labels_df = pd.read_csv(args.labels, sep="\t")
    bio_df = pd.read_csv(args.bio_features, sep="\t")
    agent_df = pd.read_csv(args.agent_attrs)

    # Use a set of tasks for multi-task reward
    tasks = [c for c in labels_df.columns if c.startswith("task_")][:5]
    print(f"  Tasks: {tasks}")

    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    agent_attrs_dict = agent_df.set_index("Gene_Symbol").to_dict("index")

    # Gene list: intersection of all sources
    genes_all = sorted(
        set(g2s.keys())
        & set(labels_df["Gene_Symbol"].astype(str))
        & set(bio_df["Gene_Symbol"].astype(str))
        & set(agent_df["Gene_Symbol"].astype(str))
    )
    print(f"  Total genes: {len(genes_all)}")

    # Build bio feature matrix (n_genes, 482) — vectorized
    bio_df = bio_df.set_index("Gene_Symbol")
    bio_cols = [c for c in bio_df.columns]
    print(f"  Bio features: {len(bio_cols)} dimensions")

    # Convert all columns to numeric, fill NaN with 0
    bio_df = bio_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    bio_all = bio_df.reindex(genes_all).values.astype(float)
    bio_all = np.nan_to_num(bio_all, nan=0.0)
    print(f"  Bio matrix: {bio_all.shape}")

    # ── Select labeled subset (positives + sampled negatives) ────────────
    task = args.task_name
    y_all = labels_df.set_index("Gene_Symbol").reindex(genes_all)[task].values
    pos_idx = np.where(y_all == 1)[0]
    neg_idx = np.where(y_all == 0)[0]
    print(f"  Task '{task}': {len(pos_idx)} positive, {len(neg_idx)} negative")

    # Subset: sample positives + negatives to roughly n_subset, 70/30 pos/neg if possible
    np.random.seed(42)
    target_pos = min(len(pos_idx), max(int(args.n_subset * 0.7), 200))
    target_neg = min(len(neg_idx), args.n_subset - target_pos)
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
    print(f"  Subset: {len(subset_genes)} genes ({subset_y.sum()} pos, {len(subset_y)-subset_y.sum()} neg)")

    # Baseline reward: LR on bio only
    bio_baseline = compute_adj_f1(subset_bio, subset_y)
    print(f"  Bio-only Adj F1 on subset: {bio_baseline*100:.2f}%")

    # ── Load model ───────────────────────────────────────────────────────
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
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Sanity test ──────────────────────────────────────────────────────
    print("\n[SANITY] Testing on 2 genes...")
    test_genes = subset_genes[:2]
    test_prompts = [make_prompt(g, build_gene_profile(g, g2s, agent_attrs_dict), tokenizer)
                    for g in test_genes]
    model.eval()
    texts, _ = generate_batch(model, tokenizer, test_prompts, device, temperature=0.3, max_tokens=250)
    for g, t in zip(test_genes, texts):
        attrs, reasoning = extract_attrs(t)
        print(f"  [{g}] attrs={len(attrs)}, reasoning={'yes' if reasoning else 'no'}, {len(t)} chars")

    # ── Initialize subset attrs matrix with 0.5 ──────────────────────────
    # Each gene has a current "best attrs vector" of dim 20.
    llm_mat = np.full((len(subset_genes), len(ATTR_VOCAB)), 0.5)
    gene2idx = {g: i for i, g in enumerate(subset_genes)}

    # Initial reward with 0.5 filler
    X_init = np.concatenate([subset_bio, llm_mat], axis=1)
    init_reward = compute_adj_f1(X_init, subset_y)
    print(f"  Initial reward (bio + filler LLM): {init_reward*100:.2f}%")

    # ── RL ───────────────────────────────────────────────────────────────
    print(f"\n[RL] GRPO: {args.rl_steps} steps, B={args.genes_per_step}, G={args.group_size}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.rl_lr
    )

    log = []
    best_reward = bio_baseline

    for step in range(args.rl_steps):
        t0 = time.time()
        # Sample batch of genes from subset
        batch_idxs = np.random.choice(len(subset_genes), size=args.genes_per_step, replace=False)
        batch_genes_step = [subset_genes[i] for i in batch_idxs]

        # Build prompts for all rollouts
        all_prompts = []
        key_list = []
        for gi_batch in batch_idxs:
            gene = subset_genes[gi_batch]
            profile = build_gene_profile(gene, g2s, agent_attrs_dict)
            prompt = make_prompt(gene, profile, tokenizer)
            for k in range(args.group_size):
                all_prompts.append(prompt)
                key_list.append(gi_batch)

        # Batch generate
        model.eval()
        texts, gen_ids_list = generate_batch(
            model, tokenizer, all_prompts, device, args.temperature, max_tokens=250
        )

        rollouts = []
        for gi_batch, text, gen_ids in zip(key_list, texts, gen_ids_list):
            attrs, reasoning = extract_attrs(text)
            rollouts.append((gi_batch, attrs, reasoning, gen_ids))

        # Compute rewards: substitute each rollout's attrs into llm_mat, compute Adj F1
        rewards = []
        for (gi_batch, attrs, reasoning, _) in rollouts:
            orig_row = llm_mat[gi_batch].copy()
            if attrs:
                llm_mat[gi_batch] = attrs_to_vec(attrs)
            X = np.concatenate([subset_bio, llm_mat], axis=1)
            r = compute_adj_f1(X, subset_y)
            llm_mat[gi_batch] = orig_row  # restore

            # Penalty for no reasoning
            if not reasoning or len(reasoning) < 30:
                r = r * 0.5
            rewards.append(r)

        rewards = np.array(rewards)

        # Advantages (GRPO: group-centered)
        advantages = np.zeros_like(rewards)
        for i, gi_batch in enumerate(batch_idxs):
            start = i * args.group_size
            group = rewards[start:start+args.group_size]
            advantages[start:start+args.group_size] = group - group.mean()

        # Normalize by global std
        adv_std = advantages.std() + 1e-8
        advantages = advantages / adv_std

        # Policy update
        model.train()
        optimizer.zero_grad()
        total_pg_loss = 0.0
        n_valid = 0

        for i, (gi_batch, attrs, reasoning, gen_ids) in enumerate(rollouts):
            adv = advantages[i]
            if abs(adv) < 1e-6 or len(gen_ids) == 0:
                continue

            prompt = all_prompts[i]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(device)
            inp = enc["input_ids"][0]
            full_ids = torch.cat([inp, gen_ids.to(device)]).unsqueeze(0)
            labels = full_ids.clone()
            labels[0, :len(inp)] = -100

            out = model(input_ids=full_ids, labels=labels)
            pg_loss = out.loss * (-adv)
            pg_loss.backward()
            total_pg_loss += pg_loss.item()
            n_valid += 1

        if n_valid > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= n_valid
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update llm_mat with best rollout per gene
        for i, gi_batch in enumerate(batch_idxs):
            start = i * args.group_size
            group = rewards[start:start+args.group_size]
            best_j = int(group.argmax())
            best_attrs = rollouts[start + best_j][1]
            if best_attrs:
                llm_mat[gi_batch] = attrs_to_vec(best_attrs)

        # Global reward on full subset
        global_X = np.concatenate([subset_bio, llm_mat], axis=1)
        global_reward = compute_adj_f1(global_X, subset_y)

        mean_r = rewards.mean()
        valid_pct = sum(1 for _, attrs, _, _ in rollouts if attrs) / len(rollouts) * 100
        reasoning_pct = sum(1 for _, _, r, _ in rollouts if r and len(r) > 30) / len(rollouts) * 100
        dt = time.time() - t0

        log.append({
            "step": step,
            "mean_reward": float(mean_r),
            "global_reward": float(global_reward),
            "bio_baseline": float(bio_baseline),
            "valid_pct": float(valid_pct),
            "reasoning_pct": float(reasoning_pct),
            "pg_loss": float(total_pg_loss / max(n_valid, 1)),
            "time_s": dt,
        })

        if global_reward > best_reward:
            best_reward = global_reward
            model.save_pretrained(os.path.join(args.outdir, "rl_best"))
            tokenizer.save_pretrained(os.path.join(args.outdir, "rl_best"))

        if step % 2 == 0:
            print(f"  Step {step:3d}: rollout_r={mean_r*100:.2f}%, global_r={global_reward*100:.2f}% "
                  f"(bio={bio_baseline*100:.2f}%), valid={valid_pct:.0f}%, "
                  f"reas={reasoning_pct:.0f}%, loss={total_pg_loss/max(n_valid,1):.4f}, {dt:.0f}s")

        if step % 10 == 0:
            pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    print(f"\n[DONE] Bio-only: {bio_baseline*100:.2f}%")
    print(f"       Best RL:  {best_reward*100:.2f}%")
    print(f"       Improvement: +{(best_reward - bio_baseline)*100:.2f}%")

    # Save final llm_mat
    np.save(os.path.join(args.outdir, "llm_attrs_subset.npy"), llm_mat)
    pd.DataFrame(llm_mat, columns=ATTR_VOCAB, index=subset_genes).to_csv(
        os.path.join(args.outdir, "llm_attrs_subset.csv")
    )
    print(f"       Saved to {args.outdir}/")


if __name__ == "__main__":
    main()
