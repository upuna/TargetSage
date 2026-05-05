#!/usr/bin/env python3
"""
Reward-Guided Semantic Attribute Optimization via GRPO (v2).

Fixes: pre-compute full attribute matrix, swap batch genes per rollout,
compute global Adjusted F1 reward on full genome.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u scripts/rl_grpo_v2.py
"""

import os, sys, json, re, argparse, time, copy
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── Prompts ──────────────────────────────────────────────────────────────────
SYS = ("You are a drug discovery scientist. Given a gene and its summary, "
       "output a JSON with therapeutic attribute scores in [0.0, 1.0]. "
       "Include any attributes relevant for target identification. "
       "Return STRICT JSON only.")

def make_prompt(gene, summary, tokenizer):
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Gene: {gene}\nSummary: {summary}\nReturn JSON:"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def extract_attrs(text):
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group())
        return {k: float(v) for k, v in obj.items()
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}

# ── Reward ───────────────────────────────────────────────────────────────────
def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))

def compute_reward(attr_matrix, genes, labels_df, tasks, attr_keys):
    """Global Adjusted F1 from full attribute matrix."""
    if len(attr_keys) == 0:
        return 0.0
    gene2idx = {g: i for i, g in enumerate(genes)}
    X = StandardScaler().fit_transform(attr_matrix)
    adj_f1s = []
    for t in tasks:
        y = labels_df.set_index("Gene_Symbol").reindex(genes)[t].values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < 20 or y[valid].sum() < 5:
            continue
        try:
            lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                    solver="lbfgs", random_state=42)
            lr.fit(X[valid], y[valid].astype(int))
            prob = lr.predict_proba(X[valid])[:, 1]
            adj_f1s.append(adjusted_f1(prob, y[valid].astype(int)))
        except Exception:
            pass
    return float(np.mean(adj_f1s)) if adj_f1s else 0.0

def generate_single(model, tokenizer, prompt, device, temperature=0.8, max_tokens=200):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_tokens, temperature=temperature,
            top_k=20, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True), enc["input_ids"][0], gen_ids

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--teacher_scores", default="data/features_llm_structured_scores.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results/rl_grpo_v2")
    ap.add_argument("--sft_epochs", type=int, default=2)
    ap.add_argument("--skip_sft", action="store_true")
    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--rl_lr", type=float, default=5e-5)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--genes_per_step", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ────────────────────────────────────────────────────────
    print("[DATA] Loading...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    teacher = pd.read_csv(args.teacher_scores)
    labels_df = pd.read_csv(args.labels, sep="\t")
    tasks = [c for c in labels_df.columns if c.startswith("task_")][:8]

    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    teacher_genes = set(teacher["Gene_Symbol"].astype(str))
    genes = sorted(set(g2s.keys()) & teacher_genes)
    print(f"  {len(genes)} genes, {len(tasks)} tasks")

    # ── Load model ───────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"[MODEL] Loading {args.model}...")
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

    # ── Phase 1: SFT ────────────────────────────────────────────────────
    sft_ckpt = os.path.join(args.outdir, "sft_checkpoint")
    if args.skip_sft and os.path.isdir(sft_ckpt):
        print(f"\n[SFT] Loading existing checkpoint from {sft_ckpt}")
        from peft import PeftModel
        model = model.base_model.model  # unwrap LoRA
        model = PeftModel.from_pretrained(model, sft_ckpt)
        tokenizer = AutoTokenizer.from_pretrained(sft_ckpt, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        print("  Loaded.")
    if not args.skip_sft:
        print("\n[SFT] Training...")
        attr_cols = [c for c in teacher.columns if c != "Gene_Symbol"]
        teacher_dict = teacher.set_index("Gene_Symbol").to_dict("index")

        train_texts = []
        for gene in genes:
            row = teacher_dict.get(gene, {})
            target = {k: round(float(v), 2) for k, v in row.items() if pd.notna(v)}
            if not target:
                continue
            prompt = make_prompt(gene, g2s[gene], tokenizer)
            train_texts.append(prompt + json.dumps(target) + tokenizer.eos_token)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01
        )
        model.train()
        for epoch in range(args.sft_epochs):
            np.random.shuffle(train_texts)
            total_loss, n = 0, 0
            for i in range(0, len(train_texts), 4):
                batch = train_texts[i:i+4]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
                labels = enc["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100
                out = model(**enc, labels=labels)
                (out.loss / 4).backward()
                if (i // 4 + 1) % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += out.loss.item()
                n += 1
                if n % 500 == 0:
                    print(f"  Epoch {epoch+1}, batch {n}: loss={total_loss/n:.4f}")
            print(f"  [SFT] Epoch {epoch+1} done, loss={total_loss/n:.4f}")

        model.save_pretrained(sft_ckpt)
        tokenizer.save_pretrained(sft_ckpt)

    # ── Pre-generate baseline attributes for ALL genes ───────────────────
    print("\n[BASELINE] Generating baseline attributes for all genes...")
    model.eval()
    baseline_attrs = {}  # gene -> {attr: score}
    for i, gene in enumerate(genes):
        prompt = make_prompt(gene, g2s[gene], tokenizer)
        text, _, _ = generate_single(model, tokenizer, prompt, device, temperature=0.1)
        baseline_attrs[gene] = extract_attrs(text)
        if (i+1) % 1000 == 0:
            valid = sum(1 for v in baseline_attrs.values() if v)
            print(f"  {i+1}/{len(genes)} done, {valid} valid")

    # Build baseline attribute matrix
    all_attr_keys = sorted(set(k for v in baseline_attrs.values() for k in v))
    print(f"  Baseline: {len(all_attr_keys)} unique attributes")

    def build_matrix(attrs_dict, keys):
        mat = np.full((len(genes), len(keys)), 0.5)
        for i, gene in enumerate(genes):
            for j, k in enumerate(keys):
                mat[i, j] = attrs_dict.get(gene, {}).get(k, 0.5)
        return mat

    baseline_matrix = build_matrix(baseline_attrs, all_attr_keys)
    baseline_reward = compute_reward(baseline_matrix, genes, labels_df, tasks, all_attr_keys)
    print(f"  Baseline reward (Adj F1): {baseline_reward:.4f}")

    # ── Phase 2: RL (GRPO) ──────────────────────────────────────────────
    print(f"\n[RL] Starting GRPO: {args.rl_steps} steps, G={args.group_size}, B={args.genes_per_step}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.rl_lr
    )

    gene2idx = {g: i for i, g in enumerate(genes)}
    log = []
    best_reward = baseline_reward

    for step in range(args.rl_steps):
        t0 = time.time()
        batch_genes = list(np.random.choice(genes, size=args.genes_per_step, replace=False))

        # Generate G rollouts per gene
        model.eval()
        rollouts = []  # (gene, attrs, inp_ids, gen_ids)
        for gene in batch_genes:
            prompt = make_prompt(gene, g2s[gene], tokenizer)
            for _ in range(args.group_size):
                text, inp_ids, gen_ids = generate_single(
                    model, tokenizer, prompt, device, args.temperature
                )
                attrs = extract_attrs(text)
                rollouts.append((gene, attrs, inp_ids, gen_ids))

        # Compute rewards: for each rollout, swap its gene's attrs into the matrix
        rewards = []
        for gene, attrs, _, _ in rollouts:
            # Copy baseline and swap this gene's attrs
            modified = baseline_attrs.copy()
            modified[gene] = attrs if attrs else baseline_attrs.get(gene, {})

            # Merge keys
            cur_keys = sorted(set(k for v in modified.values() for k in v))
            mat = build_matrix(modified, cur_keys)
            r = compute_reward(mat, genes, labels_df, tasks, cur_keys)
            rewards.append(r)

        rewards = np.array(rewards)

        # Advantages (GRPO: group-centered, batch-normalized)
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

        # Policy gradient update
        model.train()
        optimizer.zero_grad()
        total_pg_loss = 0.0
        n_valid = 0

        for i, (gene, attrs, inp_ids, gen_ids) in enumerate(rollouts):
            adv = advantages[i]
            if abs(adv) < 1e-8 or len(gen_ids) == 0:
                continue

            full_ids = torch.cat([inp_ids, gen_ids]).unsqueeze(0).to(device)
            labels = full_ids.clone()
            labels[0, :len(inp_ids)] = -100

            out = model(input_ids=full_ids, labels=labels)
            # REINFORCE: loss = -advantage * log_prob = advantage * CE_loss
            pg_loss = out.loss * (-adv)
            pg_loss.backward()
            total_pg_loss += pg_loss.item()
            n_valid += 1

        if n_valid > 0:
            # Scale gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= n_valid
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update baseline attrs for batch genes (use best rollout)
        idx = 0
        for gene in batch_genes:
            gene_rs = rewards[idx:idx+args.group_size]
            best_j = gene_rs.argmax()
            best_attrs = rollouts[idx + best_j][1]
            if best_attrs:
                baseline_attrs[gene] = best_attrs
            idx += args.group_size

        # Rebuild keys periodically
        if step % 10 == 0:
            all_attr_keys = sorted(set(k for v in baseline_attrs.values() for k in v))
            baseline_matrix = build_matrix(baseline_attrs, all_attr_keys)

        mean_r = rewards.mean()
        n_attrs = len(set(k for _, attrs, _, _ in rollouts for k in attrs))
        valid_pct = sum(1 for _, attrs, _, _ in rollouts if attrs) / len(rollouts) * 100
        dt = time.time() - t0

        log.append({
            "step": step, "mean_reward": float(mean_r),
            "best_reward": float(rewards.max()),
            "n_unique_attrs": n_attrs, "valid_pct": float(valid_pct),
            "pg_loss": float(total_pg_loss / max(n_valid, 1)),
            "time_s": dt,
        })

        if mean_r > best_reward:
            best_reward = mean_r
            model.save_pretrained(os.path.join(args.outdir, "rl_best"))
            tokenizer.save_pretrained(os.path.join(args.outdir, "rl_best"))

        if step % 5 == 0:
            print(f"  Step {step:3d}: reward={mean_r:.4f}±{rewards.std():.4f}, "
                  f"best={rewards.max():.4f}, attrs={n_attrs}, "
                  f"valid={valid_pct:.0f}%, loss={total_pg_loss/max(n_valid,1):.4f}, {dt:.0f}s")

    # Save log
    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    # Final: regenerate all genes with best model
    print("\n[FINAL] Generating RL-optimized attributes...")
    model.eval()
    final_attrs = {}
    for i, gene in enumerate(genes):
        prompt = make_prompt(gene, g2s[gene], tokenizer)
        text, _, _ = generate_single(model, tokenizer, prompt, device, temperature=0.1)
        final_attrs[gene] = extract_attrs(text)
        if (i+1) % 1000 == 0:
            print(f"  {i+1}/{len(genes)}")

    final_keys = sorted(set(k for v in final_attrs.values() for k in v))
    rows = [{"Gene_Symbol": g, **{k: final_attrs[g].get(k, np.nan) for k in final_keys}} for g in genes]
    out_csv = os.path.join(args.outdir, "features_rl_attributes.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    final_mat = build_matrix(final_attrs, final_keys)
    final_reward = compute_reward(final_mat, genes, labels_df, tasks, final_keys)
    print(f"\n[DONE] Baseline Adj-F1: {baseline_reward:.4f} → RL Adj-F1: {final_reward:.4f}")
    print(f"  {len(final_keys)} attribute dimensions discovered")
    print(f"  Saved to {out_csv}")

if __name__ == "__main__":
    main()
