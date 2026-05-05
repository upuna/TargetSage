#!/usr/bin/env python3
"""
Direct GRPO v2: task-specific reasoning module optimized by RL.

Key optimizations:
- Skip baseline generation: use agent_attrs (28-dim) as baseline matrix for ALL genes.
- Only generate LLM rollouts for batch genes during RL steps.
- Few-shot prompt for reliable JSON format output.
- Batch generation (multiple rollouts in one forward pass).

Usage:
  CUDA_VISIBLE_DEVICES=1 python -u scripts/rl_grpo_direct.py
"""

import os, sys, json, re, argparse, time
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── Few-shot prompt ──────────────────────────────────────────────────────────
# Fixed 20-dimensional therapeutic attribute vocabulary
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
    "Given a gene's evidence profile, produce a brief biological reasoning chain followed by "
    f"a JSON object scoring these {len(ATTR_VOCAB)} therapeutic attributes in [0.0, 1.0]: "
    f"{ATTR_LIST_STR}. "
    "Always include REASONING: followed by ATTRIBUTES: sections. "
    "Reasoning should be 2-3 sentences explaining biological relevance. "
    "Include all attributes in the JSON."
)

FEW_SHOT = [
    {
        "user": "Gene: BRCA1\nEvidence: Breast cancer susceptibility gene; DNA repair; pLI=1.00 (highly constrained); GWAS: breast cancer, ovarian cancer; tractable small molecule and antibody.",
        "assistant": (
            "REASONING: BRCA1 is a tumor suppressor with strong genetic constraint (pLI=1.00) and direct GWAS linkage to breast/ovarian cancer. "
            "Its role in homologous recombination DNA repair makes it a validated target for PARP inhibitor synthetic lethality. "
            "Multiple tractability modalities are available.\n"
            "ATTRIBUTES: {\"disease_association\": 0.95, \"druggability\": 0.80, \"loss_of_function_tolerance\": 0.05, \"safety_profile\": 0.60, \"tissue_specificity\": 0.30, \"cancer_relevance\": 0.95, \"tractable_small_molecule\": 0.85, \"tractable_antibody\": 0.50, \"tractable_protac\": 0.40, \"enzyme_activity\": 0.70, \"membrane_localization\": 0.10, \"secreted_protein\": 0.05, \"pathway_centrality\": 0.85, \"gwas_support\": 0.95, \"essential_gene\": 0.60, \"novelty\": 0.20, \"expression_breadth\": 0.70, \"protein_interactions\": 0.80, \"biomarker_potential\": 0.85, \"clinical_validation\": 0.90}"
        ),
    },
    {
        "user": "Gene: ACLY\nEvidence: ATP-citrate lyase; lipid metabolism; pLI=0.92; essential in some cancer cell lines; small molecule inhibitor bempedoic acid FDA approved.",
        "assistant": (
            "REASONING: ACLY is a metabolic enzyme with validated small molecule tractability through bempedoic acid. "
            "Its role in cytosolic acetyl-CoA generation links it to both lipid biosynthesis and cancer metabolism, with moderate loss-of-function constraint indicating functional importance.\n"
            "ATTRIBUTES: {\"disease_association\": 0.75, \"druggability\": 0.95, \"loss_of_function_tolerance\": 0.08, \"safety_profile\": 0.75, \"tissue_specificity\": 0.40, \"cancer_relevance\": 0.70, \"tractable_small_molecule\": 0.98, \"tractable_antibody\": 0.10, \"tractable_protac\": 0.20, \"enzyme_activity\": 0.95, \"membrane_localization\": 0.05, \"secreted_protein\": 0.05, \"pathway_centrality\": 0.80, \"gwas_support\": 0.60, \"essential_gene\": 0.40, \"novelty\": 0.30, \"expression_breadth\": 0.85, \"protein_interactions\": 0.60, \"biomarker_potential\": 0.60, \"clinical_validation\": 0.90}"
        ),
    },
]


def make_prompt(gene, profile, tokenizer):
    """Build few-shot prompt from enriched profile."""
    msgs = [{"role": "system", "content": SYS}]
    for ex in FEW_SHOT:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": f"Gene: {gene}\nEvidence: {profile}"})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_attrs(text):
    """Extract attribute scores (flexible: JSON or key:value / key=value) and reasoning."""
    attrs = {}

    # Try JSON first
    json_str = None
    m = re.search(r'ATTRIBUTES:\s*(\{[^{}]*\})', text, re.DOTALL | re.IGNORECASE)
    if m:
        json_str = m.group(1)
    else:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            json_str = m.group(0)

    if json_str:
        try:
            obj = json.loads(json_str)
            for k, v in obj.items():
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1:
                    attrs[k] = float(v)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback: match key:value or key=value patterns for ATTR_VOCAB names
    if len(attrs) < 5:
        for key in ATTR_VOCAB:
            # Match key: 0.xx or key= 0.xx or "key": 0.xx
            pat = rf'["\s]*{re.escape(key)}["\s]*[:=]\s*([0-9]*\.?[0-9]+)'
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    v = float(m.group(1))
                    if 0 <= v <= 1:
                        attrs[key] = v
                except ValueError:
                    pass

    # Extract reasoning
    reasoning = ""
    rm = re.search(r'REASONING:\s*(.*?)(?:ATTRIBUTES:|$)', text, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()[:500]
    return attrs, reasoning


# ── Reward ───────────────────────────────────────────────────────────────────
def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


def compute_reward(attr_matrix, genes, labels_df, tasks):
    """Global Adjusted F1 from full attribute matrix (training-valid rows only)."""
    X = StandardScaler().fit_transform(attr_matrix)
    adj_f1s = []
    for t in tasks:
        y = labels_df.set_index("Gene_Symbol").reindex(genes)[t].values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < 20 or y[valid].sum() < 5:
            continue
        try:
            lr = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced",
                                    solver="lbfgs", random_state=42)
            lr.fit(X[valid], y[valid].astype(int))
            prob = lr.predict_proba(X[valid])[:, 1]
            adj_f1s.append(adjusted_f1(prob, y[valid].astype(int)))
        except Exception:
            pass
    return float(np.mean(adj_f1s)) if adj_f1s else 0.0


def generate_batch(model, tokenizer, prompts, device, temperature=0.8, max_tokens=250):
    """Batch-generate multiple prompts at once."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_tokens, temperature=temperature,
            top_k=20, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    # Extract generated portions
    texts = []
    gen_ids_list = []
    inp_len = enc["input_ids"].shape[1]
    for i in range(len(prompts)):
        gen_ids = out[i, inp_len:]
        # Remove padding
        gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
        gen_ids_list.append(gen_ids)
    return texts, enc["input_ids"], gen_ids_list


def build_gene_profile(gene, g2s, agent_attrs_dict, max_len=800):
    """Build compact enriched profile string."""
    summary = g2s.get(gene, "")[:400]
    parts = [summary] if summary else []

    if agent_attrs_dict and gene in agent_attrs_dict:
        agent_info = agent_attrs_dict[gene]
        top_attrs = []
        for k, v in sorted(agent_info.items()):
            if pd.notna(v) and v != 0:
                if isinstance(v, (int, float)):
                    top_attrs.append(f"{k}={v:.2f}")
                else:
                    top_attrs.append(f"{k}={v}")
            if len(top_attrs) >= 12:
                break
        if top_attrs:
            parts.append("; ".join(top_attrs))

    profile = " | ".join(parts)[:max_len]
    return profile if profile else "No evidence available."


def agent_matrix_from_df(agent_df, genes):
    """Build baseline attribute matrix from agent-extracted attributes."""
    attr_cols = [c for c in agent_df.columns if c != "Gene_Symbol"]
    agent_df = agent_df.set_index("Gene_Symbol")
    # Fill NaN with 0
    mat = np.zeros((len(genes), len(attr_cols)))
    for i, g in enumerate(genes):
        if g in agent_df.index:
            row = agent_df.loc[g]
            for j, c in enumerate(attr_cols):
                v = row[c]
                # Convert non-numeric to 0
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        mat[i, j] = fv
                except (ValueError, TypeError):
                    mat[i, j] = 0.0
    return mat, attr_cols


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--agent_attrs", default="results/agent_tools/agent_attributes_filtered.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results/rl_grpo_direct")
    ap.add_argument("--rl_steps", type=int, default=200)
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
    tasks = [c for c in labels_df.columns if c.startswith("task_")][:8]

    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))

    print(f"  Loading agent attributes from {args.agent_attrs}")
    agent_df = pd.read_csv(args.agent_attrs)
    agent_attrs_dict = agent_df.set_index("Gene_Symbol").to_dict("index")

    genes = sorted(set(g2s.keys()) & set(labels_df["Gene_Symbol"].astype(str)) & set(agent_df["Gene_Symbol"].astype(str)))
    print(f"  {len(genes)} genes, {len(tasks)} tasks")

    # Build baseline matrix from agent attributes (no LLM generation needed!)
    print("[BASELINE] Building baseline matrix from agent attributes...")
    baseline_mat, baseline_cols = agent_matrix_from_df(agent_df, genes)
    print(f"  Baseline matrix: {baseline_mat.shape}")

    baseline_reward = compute_reward(baseline_mat, genes, labels_df, tasks)
    print(f"  Baseline reward (Adj F1): {baseline_reward:.4f}")

    # Gene -> row index in baseline matrix
    gene2idx = {g: i for i, g in enumerate(genes)}

    # ── Load model ───────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"\n[MODEL] Loading {args.model} (direct RL, no SFT)...")
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

    # ── Quick sanity test ────────────────────────────────────────────────
    print("\n[SANITY] Testing generation on 2 genes...")
    test_genes = genes[:2]
    test_prompts = [make_prompt(g, build_gene_profile(g, g2s, agent_attrs_dict), tokenizer) for g in test_genes]
    model.eval()
    texts, _, _ = generate_batch(model, tokenizer, test_prompts, device, temperature=0.5, max_tokens=200)
    for g, t in zip(test_genes, texts):
        attrs, reasoning = extract_attrs(t)
        print(f"  [{g}] attrs={len(attrs)}, reasoning={'yes' if reasoning else 'no'}")
        print(f"    sample: {t[:200]}")

    # ── RL (GRPO) ────────────────────────────────────────────────────────
    print(f"\n[RL] Starting GRPO: {args.rl_steps} steps, G={args.group_size}, B={args.genes_per_step}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.rl_lr
    )

    # Store accumulated LLM attributes (will replace agent attrs during training)
    llm_attrs = {}  # gene -> {attr_name: score}
    llm_attr_keys = set()
    log = []
    best_reward = baseline_reward

    for step in range(args.rl_steps):
        t0 = time.time()
        batch_genes = list(np.random.choice(genes, size=args.genes_per_step, replace=False))

        # Build prompts for all (gene, rollout) pairs
        all_prompts = []
        gene_rollout_keys = []
        for gene in batch_genes:
            profile = build_gene_profile(gene, g2s, agent_attrs_dict)
            prompt = make_prompt(gene, profile, tokenizer)
            for k in range(args.group_size):
                all_prompts.append(prompt)
                gene_rollout_keys.append((gene, k))

        # Batch generate
        model.eval()
        texts, inp_ids_batch, gen_ids_list = generate_batch(
            model, tokenizer, all_prompts, device, args.temperature, max_tokens=200
        )

        rollouts = []
        for (gene, k), text, gen_ids in zip(gene_rollout_keys, texts, gen_ids_list):
            attrs, reasoning = extract_attrs(text)
            rollouts.append((gene, attrs, reasoning, gen_ids))

        # Compute rewards using FIXED 20-dim vocab
        # Maintain llm_mat: (n_genes, 20) where each row is the LLM's scores
        # Initially filled with 0.5, updated as llm_attrs accumulates
        if step == 0:
            llm_mat = np.full((len(genes), len(ATTR_VOCAB)), 0.5)
            # Persist llm_mat across steps
            globals()['_llm_mat'] = llm_mat
        llm_mat = globals()['_llm_mat']

        rewards = []
        for (gene, attrs, reasoning, _) in rollouts:
            # Build candidate row from attrs (missing keys = 0.5)
            gi = gene2idx.get(gene)
            if gi is None:
                rewards.append(0.0)
                continue

            cand_row = llm_mat[gi].copy()
            for kj, kname in enumerate(ATTR_VOCAB):
                if kname in attrs:
                    cand_row[kj] = attrs[kname]

            # Substitute this row in llm_mat for reward computation
            orig_row = llm_mat[gi].copy()
            llm_mat[gi] = cand_row
            full_mat = np.concatenate([baseline_mat, llm_mat], axis=1)
            r = compute_reward(full_mat, genes, labels_df, tasks)
            llm_mat[gi] = orig_row  # restore

            # Hard requirement: must have reasoning to get full reward
            if not reasoning or len(reasoning) < 30:
                r = r * 0.3  # heavy penalty but not zero (to preserve signal)
            # Bonus proportional to how many vocab attrs were scored
            if attrs:
                coverage = sum(1 for k in ATTR_VOCAB if k in attrs) / len(ATTR_VOCAB)
                r = r * (0.7 + 0.3 * coverage)

            rewards.append(r)

        rewards = np.array(rewards)

        # Advantages (GRPO)
        batch_std = rewards.std() + 1e-8
        advantages = []
        idx = 0
        for gene in batch_genes:
            gene_rs = rewards[idx:idx+args.group_size]
            gene_mean = gene_rs.mean()
            for r in gene_rs:
                advantages.append((r - gene_mean) / batch_std)
            idx += args.group_size
        advantages = np.array(advantages)

        # Policy update (process one rollout at a time to avoid memory issues)
        model.train()
        optimizer.zero_grad()
        total_pg_loss = 0.0
        n_valid = 0

        for i, (gene, attrs, reasoning, gen_ids) in enumerate(rollouts):
            adv = advantages[i]
            if abs(adv) < 1e-8 or len(gen_ids) == 0:
                continue

            # Reconstruct full input for this rollout
            prompt = all_prompts[i]
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
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
        idx = 0
        for gene in batch_genes:
            gene_rs = rewards[idx:idx+args.group_size]
            best_j = int(gene_rs.argmax())
            best_attrs = rollouts[idx + best_j][1]
            gi = gene2idx.get(gene)
            if best_attrs and gi is not None:
                llm_attrs[gene] = best_attrs
                for kj, kname in enumerate(ATTR_VOCAB):
                    if kname in best_attrs:
                        llm_mat[gi, kj] = best_attrs[kname]
            idx += args.group_size

        mean_r = rewards.mean()
        valid_pct = sum(1 for _, attrs, _, _ in rollouts if attrs) / len(rollouts) * 100
        reasoning_pct = sum(1 for _, _, r, _ in rollouts if r and len(r) > 30) / len(rollouts) * 100
        dt = time.time() - t0

        log.append({
            "step": step, "mean_reward": float(mean_r),
            "best_reward": float(rewards.max()),
            "valid_pct": float(valid_pct),
            "reasoning_pct": float(reasoning_pct),
            "pg_loss": float(total_pg_loss / max(n_valid, 1)),
            "time_s": dt,
            "n_llm_keys": len(llm_attr_keys),
        })

        if mean_r > best_reward:
            best_reward = mean_r
            model.save_pretrained(os.path.join(args.outdir, "rl_best"))
            tokenizer.save_pretrained(os.path.join(args.outdir, "rl_best"))

        if step % 2 == 0:
            print(f"  Step {step:3d}: reward={mean_r:.4f}+/-{rewards.std():.4f}, "
                  f"best={rewards.max():.4f}, valid={valid_pct:.0f}%, "
                  f"reasoning={reasoning_pct:.0f}%, keys={len(llm_attr_keys)}, "
                  f"loss={total_pg_loss/max(n_valid,1):.4f}, {dt:.0f}s")

        # Save log periodically
        if step % 10 == 0:
            pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)
    print(f"\n[DONE] Baseline reward: {baseline_reward:.4f}, Best RL reward: {best_reward:.4f}")
    print(f"  Checkpoint: {os.path.join(args.outdir, 'rl_best')}")


if __name__ == "__main__":
    main()
