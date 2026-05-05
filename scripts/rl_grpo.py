#!/usr/bin/env python3
"""
Reward-Guided Semantic Attribute Optimization via GRPO.

End-to-end: SFT → RL on Qwen2.5-1.5B-Instruct with LoRA.
Uses AUROC proxy reward computed from extracted attributes vs known target labels.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/rl_grpo.py
"""

import os, sys, json, re, argparse, time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── Prompts ──────────────────────────────────────────────────────────────────
SYS = ("You are a drug discovery scientist. Given a gene and its summary, "
       "output a JSON with therapeutic attribute scores in [0.0, 1.0]. "
       "You may include any attributes relevant for target identification. "
       "Return STRICT JSON only.")

def make_prompt(gene, summary, tokenizer):
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Gene: {gene}\nSummary: {summary}\n\nReturn JSON:"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_attrs(text):
    """Extract {attr: score} dict from LLM output."""
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group())
        return {k: float(v) for k, v in obj.items()
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1}
    except (json.JSONDecodeError, ValueError):
        return {}


# ── Reward ───────────────────────────────────────────────────────────────────
class ProxyReward:
    """Fast Adjusted-F1-based reward from attribute vectors vs target labels."""

    def __init__(self, labels_path, tasks=None):
        df = pd.read_csv(labels_path, sep="\t")
        self.genes = df["Gene_Symbol"].astype(str).tolist()
        self.gene2idx = {g: i for i, g in enumerate(self.genes)}
        if tasks is None:
            tasks = [c for c in df.columns if c.startswith("task_")][:5]  # top 5 for speed
        self.tasks = tasks
        self.Y = {}
        for t in tasks:
            y = df[t].values.astype(float)
            self.Y[t] = y

    @staticmethod
    def adjusted_f1(probs, y):
        """Compute Adjusted F1: R_soft^2 / p_bar."""
        pos_mask = y == 1
        if pos_mask.sum() == 0:
            return 0.0
        R_soft = probs[pos_mask].mean()
        p_bar = probs.mean()
        if p_bar < 1e-10:
            return 0.0
        return float(R_soft ** 2 / p_bar)

    def __call__(self, gene_attrs_list):
        """
        gene_attrs_list: list of (gene_symbol, {attr: score}) tuples from one rollout.
        Returns scalar reward (macro-averaged Adjusted F1).
        """
        if not gene_attrs_list:
            return 0.0

        # Collect all attribute keys
        all_keys = set()
        for _, attrs in gene_attrs_list:
            all_keys.update(attrs.keys())
        if not all_keys:
            return 0.0
        keys = sorted(all_keys)

        # Build attribute matrix for genes in this batch
        n = len(self.genes)
        X = np.full((n, len(keys)), 0.5)  # default 0.5
        for gene, attrs in gene_attrs_list:
            idx = self.gene2idx.get(gene)
            if idx is not None:
                for j, k in enumerate(keys):
                    X[idx, j] = attrs.get(k, 0.5)

        # Compute Adjusted F1 across tasks
        X_sc = StandardScaler().fit_transform(X)
        adj_f1s = []
        for t in self.tasks:
            y = self.Y[t]
            valid = ~np.isnan(y)
            if valid.sum() < 20 or y[valid].sum() < 5:
                continue
            try:
                lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                        solver="lbfgs", random_state=42)
                lr.fit(X_sc[valid], y[valid].astype(int))
                prob = lr.predict_proba(X_sc[valid])[:, 1]
                af1 = self.adjusted_f1(prob, y[valid].astype(int))
                adj_f1s.append(af1)
            except Exception:
                pass
        return float(np.mean(adj_f1s)) if adj_f1s else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--teacher_scores", default="data/features_llm_structured_scores.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results/rl_grpo")

    # SFT
    ap.add_argument("--sft_epochs", type=int, default=2)
    ap.add_argument("--sft_lr", type=float, default=2e-4)
    ap.add_argument("--skip_sft", action="store_true")

    # RL
    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--rl_lr", type=float, default=3e-5)
    ap.add_argument("--group_size", type=int, default=8, help="G: rollouts per gene")
    ap.add_argument("--genes_per_step", type=int, default=16, help="B: genes per RL step")
    ap.add_argument("--eps_low", type=float, default=0.1)
    ap.add_argument("--eps_high", type=float, default=0.2)
    ap.add_argument("--kl_beta", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load data ────────────────────────────────────────────────────────────
    print("[DATA] Loading...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    teacher = pd.read_csv(args.teacher_scores)
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))

    # genes with both summary and teacher scores
    teacher_genes = set(teacher["Gene_Symbol"].astype(str))
    genes = sorted(set(g2s.keys()) & teacher_genes)
    print(f"  {len(genes)} genes with summaries + teacher scores")

    # ── Load model ───────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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

    # ── Phase 1: SFT ────────────────────────────────────────────────────────
    sft_ckpt = os.path.join(args.outdir, "sft_checkpoint")

    if not args.skip_sft:
        print("\n[SFT] Building training data...")
        attr_cols = [c for c in teacher.columns if c != "Gene_Symbol"]
        teacher_dict = teacher.set_index("Gene_Symbol").to_dict("index")

        train_texts = []
        for gene in genes:
            summary = g2s[gene]
            row = teacher_dict.get(gene, {})
            target = {k: round(float(v), 2) for k, v in row.items() if pd.notna(v)}
            if not target:
                continue
            prompt = make_prompt(gene, summary, tokenizer)
            target_json = json.dumps(target)
            train_texts.append(prompt + target_json + tokenizer.eos_token)

        print(f"  {len(train_texts)} SFT examples")

        # Simple SFT training loop
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.sft_lr, weight_decay=0.01
        )

        model.train()
        for epoch in range(args.sft_epochs):
            np.random.shuffle(train_texts)
            total_loss = 0
            n_batches = 0
            for i in range(0, len(train_texts), 4):  # batch=4
                batch = train_texts[i:i+4]
                enc = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
                labels = enc["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100

                out = model(**enc, labels=labels)
                loss = out.loss / 4  # gradient accumulation
                loss.backward()

                if (i // 4 + 1) % 4 == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * 4
                n_batches += 1

                if n_batches % 100 == 0:
                    print(f"  Epoch {epoch+1}, batch {n_batches}: loss={total_loss/n_batches:.4f}")

            print(f"  [SFT] Epoch {epoch+1} done, avg loss={total_loss/max(n_batches,1):.4f}")

        model.save_pretrained(sft_ckpt)
        tokenizer.save_pretrained(sft_ckpt)
        print(f"  [SFT] Saved to {sft_ckpt}")
    else:
        print(f"[SFT] Skipping, loading from {sft_ckpt}")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, sft_ckpt)

    # ── Phase 2: RL (GRPO) ──────────────────────────────────────────────────
    print(f"\n[RL] Starting GRPO: {args.rl_steps} steps, G={args.group_size}, B={args.genes_per_step}")

    reward_fn = ProxyReward(args.labels)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.rl_lr, weight_decay=0.0
    )

    # Reference log-probs (from SFT model, frozen)
    ref_model = None  # use KL approx via old logprobs

    log = []
    best_reward = 0.0

    for step in range(args.rl_steps):
        t0 = time.time()

        # Sample batch of genes
        batch_genes = list(np.random.choice(genes, size=args.genes_per_step, replace=False))

        # ── Generate G rollouts per gene ─────────────────────────────────
        model.eval()
        all_rollouts = []  # [(gene, generated_text, attrs, input_ids, output_ids)]

        for gene in batch_genes:
            summary = g2s[gene]
            prompt = make_prompt(gene, summary, tokenizer)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)

            for _ in range(args.group_size):
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=20, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_ids = out[0][enc["input_ids"].shape[1]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                attrs = extract_attrs(gen_text)
                all_rollouts.append((gene, gen_text, attrs, enc["input_ids"][0], gen_ids))

        # ── Compute rewards ──────────────────────────────────────────────
        # Group rollouts by gene
        gene_rollouts = {}
        for gene, text, attrs, inp_ids, out_ids in all_rollouts:
            gene_rollouts.setdefault(gene, []).append((text, attrs, inp_ids, out_ids))

        # Compute reward per rollout
        rewards = []
        for gene in batch_genes:
            for text, attrs, inp_ids, out_ids in gene_rollouts[gene]:
                # Per-gene reward: how good are THIS gene's attributes
                r = reward_fn([(gene, attrs)])
                rewards.append(r)

        rewards = np.array(rewards)

        # ── Compute advantages (GRPO) ────────────────────────────────────
        batch_std = rewards.std() + 1e-8
        advantages = []
        idx = 0
        for gene in batch_genes:
            n = len(gene_rollouts[gene])
            gene_rewards = rewards[idx:idx+n]
            gene_mean = gene_rewards.mean()
            for r in gene_rewards:
                advantages.append((r - gene_mean) / batch_std)
            idx += n
        advantages = np.array(advantages)

        # ── Policy gradient update ───────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_updates = 0

        idx = 0
        for gene in batch_genes:
            for text, attrs, inp_ids, out_ids in gene_rollouts[gene]:
                adv = advantages[idx]
                idx += 1

                if abs(adv) < 1e-6 or len(out_ids) == 0:
                    continue

                # Compute log-prob of generated tokens under current policy
                full_ids = torch.cat([inp_ids, out_ids]).unsqueeze(0).to(device)
                labels = full_ids.clone()
                labels[0, :len(inp_ids)] = -100  # mask prompt

                out = model(input_ids=full_ids, labels=labels)
                # Weighted loss: -advantage * log_prob (REINFORCE)
                # Negative because out.loss = -log_prob already
                pg_loss = out.loss * (-adv)

                # KL penalty (approximate: penalize large loss deviations)
                kl_penalty = args.kl_beta * out.loss.detach()

                loss = pg_loss + kl_penalty
                loss = loss / (args.genes_per_step * args.group_size)
                loss.backward()

                total_loss += loss.item()
                n_updates += 1

        # Clip and step
        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────────────
        mean_r = rewards.mean()
        n_attrs = len(set(k for _, _, attrs, _, _ in all_rollouts for k in attrs))
        valid_pct = sum(1 for _, _, attrs, _, _ in all_rollouts if attrs) / len(all_rollouts) * 100
        dt = time.time() - t0

        log.append({
            "step": step, "mean_reward": float(mean_r),
            "std_reward": float(rewards.std()),
            "n_unique_attrs": n_attrs,
            "valid_pct": float(valid_pct),
            "loss": float(total_loss),
            "time_s": float(dt),
        })

        if mean_r > best_reward:
            best_reward = mean_r
            model.save_pretrained(os.path.join(args.outdir, "rl_best"))
            tokenizer.save_pretrained(os.path.join(args.outdir, "rl_best"))

        if step % 5 == 0:
            print(f"  Step {step:3d}/{args.rl_steps}: reward={mean_r:.4f}±{rewards.std():.4f}, "
                  f"attrs={n_attrs}, valid={valid_pct:.0f}%, loss={total_loss:.4f}, {dt:.1f}s")

    # ── Save results ─────────────────────────────────────────────────────
    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)

    # Generate final attributes with best model
    print("\n[EVAL] Generating RL-optimized attributes with best model...")
    model.eval()
    final_attrs = {}
    for i, gene in enumerate(genes):
        summary = g2s[gene]
        prompt = make_prompt(gene, summary, tokenizer)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                temperature=0.1, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        attrs = extract_attrs(gen)
        final_attrs[gene] = attrs
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(genes)} genes generated")

    # Save as CSV
    all_keys = sorted(set(k for v in final_attrs.values() for k in v))
    rows = []
    for gene in genes:
        row = {"Gene_Symbol": gene}
        for k in all_keys:
            row[k] = final_attrs[gene].get(k, np.nan)
        rows.append(row)
    out_csv = os.path.join(args.outdir, "features_rl_attributes.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[DONE] RL attributes saved to {out_csv}")
    print(f"  Discovered {len(all_keys)} unique attribute dimensions")
    print(f"  Best reward: {best_reward:.4f}")

    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
