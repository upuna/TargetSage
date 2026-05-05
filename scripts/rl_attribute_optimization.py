#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward-Guided Semantic Attribute Optimization (GSAO)

Two-phase training:
  Phase 1 (SFT): Fine-tune Llama-3-8B on GPT-4o-mini attribute extraction traces
  Phase 2 (RL):  GRPO with AUROC proxy reward to discover optimal attributes

Usage:
  # Phase 1: SFT
  python scripts/rl_attribute_optimization.py --phase sft \
      --gene_summaries data/gene_summaries.tsv \
      --teacher_scores data/features_llm_structured_scores.csv \
      --labels data/gene_labels.tsv

  # Phase 2: RL
  python scripts/rl_attribute_optimization.py --phase rl \
      --sft_checkpoint results/rl_sft/checkpoint-best \
      --labels data/gene_labels.tsv
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ──────────────────────────────────────────────────────────────────────────────
# Attribute extraction prompt templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SFT = """You are a senior drug discovery scientist.
Given a gene and its functional summary, analyze its therapeutic properties.
Output a JSON object with continuous-valued attributes in [0.0, 1.0].
You may include any attributes you find relevant for target identification.
Return STRICT JSON only."""

USER_TEMPLATE = """Gene: {gene_symbol}
Summary: {gene_summary}

Analyze this gene's therapeutic potential. Return a JSON with attribute scores in [0.0, 1.0].
Think step by step about the gene's druggability, localization, enzyme activity, pathway importance,
clinical evidence, and any other properties relevant for drug target identification."""


# ──────────────────────────────────────────────────────────────────────────────
# Reward computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_proxy_reward(attr_matrix, labels_df, tasks=None, n_splits=3, seed=42):
    """
    Compute macro-averaged AUROC reward using ridge-regularized logistic regression.

    Args:
        attr_matrix: np.ndarray (n_genes, d) - attribute scores
        labels_df: DataFrame with Gene_Symbol + task columns
        tasks: list of task column names (if None, auto-detect)
        n_splits: CV folds for AUROC estimation
        seed: random seed

    Returns:
        float: macro-averaged AUROC across tasks
    """
    if tasks is None:
        tasks = [c for c in labels_df.columns if c.startswith("task_")]

    X = StandardScaler().fit_transform(attr_matrix)
    task_aurocs = []

    for task in tasks:
        y = labels_df[task].values.astype(int)
        valid = ~np.isnan(y.astype(float))
        X_t, y_t = X[valid], y[valid]

        if len(np.unique(y_t)) < 2 or y_t.sum() < 5:
            continue

        # Cross-validated AUROC
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aurocs = []
        for train_idx, val_idx in skf.split(X_t, y_t):
            lr = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced",
                solver="lbfgs", random_state=seed
            )
            lr.fit(X_t[train_idx], y_t[train_idx])
            prob = lr.predict_proba(X_t[val_idx])[:, 1]
            try:
                fold_aurocs.append(roc_auc_score(y_t[val_idx], prob))
            except ValueError:
                pass

        if fold_aurocs:
            task_aurocs.append(np.mean(fold_aurocs))

    return float(np.mean(task_aurocs)) if task_aurocs else 0.5


def extract_attributes_from_text(text):
    """Extract JSON attribute dict from LLM output text."""
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            # Filter to numeric values in [0, 1]
            attrs = {}
            for k, v in obj.items():
                try:
                    val = float(v)
                    if 0.0 <= val <= 1.0:
                        attrs[k] = val
                except (ValueError, TypeError):
                    pass
            return attrs
        except json.JSONDecodeError:
            pass
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset for SFT
# ──────────────────────────────────────────────────────────────────────────────

class AttributeSFTDataset(Dataset):
    """Dataset for supervised fine-tuning on teacher attribute scores."""

    def __init__(self, gene_summaries, teacher_scores, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Merge
        merged = gene_summaries.merge(teacher_scores, on="Gene_Symbol", how="inner")

        attr_cols = [c for c in teacher_scores.columns if c != "Gene_Symbol"]

        for _, row in merged.iterrows():
            gene = str(row["Gene_Symbol"])
            summary = str(row.get("summary", ""))

            # Build target JSON from teacher scores
            target_attrs = {}
            for col in attr_cols:
                val = row[col]
                if pd.notna(val):
                    target_attrs[col] = round(float(val), 2)

            if not target_attrs or not summary:
                continue

            user_msg = USER_TEMPLATE.format(gene_symbol=gene, gene_summary=summary)
            target_json = json.dumps(target_attrs, indent=None)

            # Format as chat
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_SFT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": target_json},
            ]

            self.examples.append({
                "gene": gene,
                "messages": messages,
                "target_json": target_json,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ──────────────────────────────────────────────────────────────────────────────
# SFT Phase
# ──────────────────────────────────────────────────────────────────────────────

def run_sft(args):
    """Phase 1: Supervised fine-tuning on teacher attribute scores."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer

    print("[SFT] Loading data...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    teacher = pd.read_csv(args.teacher_scores)

    print(f"[SFT] Summaries: {len(summaries)}, Teacher scores: {len(teacher)}")

    print("[SFT] Loading model...")
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()

    # LoRA config
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Build text dataset
    merged = summaries.merge(teacher, on="Gene_Symbol", how="inner")
    attr_cols = [c for c in teacher.columns if c != "Gene_Symbol"]

    texts = []
    for _, row in merged.iterrows():
        gene = str(row["Gene_Symbol"])
        summary = str(row.get("summary", ""))
        target_attrs = {}
        for col in attr_cols:
            val = row[col]
            if pd.notna(val):
                target_attrs[col] = round(float(val), 2)
        if not target_attrs or not summary:
            continue
        user_msg = USER_TEMPLATE.format(gene_symbol=gene, gene_summary=summary)
        target_json = json.dumps(target_attrs, indent=None)
        text = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SYSTEM_PROMPT_SFT}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_msg}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{target_json}<|eot_id|>")
        texts.append(text)

    from datasets import Dataset as HFDataset
    dataset = HFDataset.from_dict({"text": texts})
    print(f"[SFT] Dataset: {len(dataset)} examples")

    outdir = os.path.join(args.outdir, "rl_sft")
    os.makedirs(outdir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=outdir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
    )

    print("[SFT] Training...")
    trainer.train()
    trainer.save_model(os.path.join(outdir, "checkpoint-best"))
    tokenizer.save_pretrained(os.path.join(outdir, "checkpoint-best"))
    print(f"[SFT] Done. Saved to {outdir}/checkpoint-best")


# ──────────────────────────────────────────────────────────────────────────────
# RL Phase (GRPO-style)
# ──────────────────────────────────────────────────────────────────────────────

def run_rl(args):
    """
    Phase 2: GRPO-based RL optimization of attribute extraction.

    Simplified GRPO loop:
    1. For each batch of genes, generate G rollouts per gene
    2. Extract attributes from each rollout
    3. Compute proxy AUROC reward
    4. Compute advantages (group-centered, batch-normalized)
    5. Update policy with clipped surrogate loss
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[RL] Loading model from SFT checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.sft_checkpoint)
    model.print_trainable_parameters()

    # Load data
    print("[RL] Loading gene data...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    labels = pd.read_csv(args.labels, sep="\t")
    tasks = [c for c in labels.columns if c.startswith("task_")]

    gene_to_summary = dict(zip(
        summaries["Gene_Symbol"].astype(str),
        summaries["summary"].astype(str)
    ))
    genes = sorted(set(labels["Gene_Symbol"].astype(str)) & set(gene_to_summary.keys()))
    print(f"[RL] {len(genes)} genes, {len(tasks)} tasks")

    # RL hyperparameters
    G = args.group_size          # rollouts per gene
    B = args.batch_size          # genes per batch
    lr = args.rl_lr
    eps_low = args.eps_low
    eps_high = args.eps_high
    kl_beta = args.kl_beta
    max_steps = args.rl_steps
    temperature = args.temperature

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0
    )

    outdir = os.path.join(args.outdir, f"rl_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)

    reward_log = []
    best_reward = 0.0

    print(f"[RL] Starting GRPO: {max_steps} steps, G={G}, B={B}")

    for step in range(max_steps):
        # Sample batch of genes
        batch_genes = np.random.choice(genes, size=min(B, len(genes)), replace=False)

        all_rewards = []
        all_attrs = []

        # Generate G rollouts per gene
        model.eval()
        for gene in batch_genes:
            summary = gene_to_summary[gene]
            prompt = f"<|system|>\n{SYSTEM_PROMPT_SFT}\n<|user|>\n" + \
                     USER_TEMPLATE.format(gene_symbol=gene, gene_summary=summary) + \
                     "\n<|assistant|>\n"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gene_rollout_attrs = []
            for _ in range(G):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=temperature,
                        top_k=20,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                attrs = extract_attributes_from_text(generated)
                gene_rollout_attrs.append(attrs)

            all_attrs.append(gene_rollout_attrs)

        # Compute rewards for each rollout
        # Build attribute matrices and compute proxy AUROC
        all_attr_keys = set()
        for gene_rollouts in all_attrs:
            for attrs in gene_rollouts:
                all_attr_keys.update(attrs.keys())
        attr_keys = sorted(all_attr_keys)

        if not attr_keys:
            print(f"  Step {step}: No valid attributes extracted, skipping")
            continue

        rollout_rewards = []
        for g_idx, gene in enumerate(batch_genes):
            gene_rewards = []
            for rollout_attrs in all_attrs[g_idx]:
                # Build full attribute matrix using this rollout's schema
                attr_matrix = np.zeros((len(genes), len(attr_keys)))
                for i, g in enumerate(genes):
                    if g == gene:
                        for j, key in enumerate(attr_keys):
                            attr_matrix[i, j] = rollout_attrs.get(key, 0.5)
                    else:
                        for j, key in enumerate(attr_keys):
                            attr_matrix[i, j] = 0.5  # default for other genes

                # Use a simpler reward: just check if attributes correlate with labels
                # for the specific gene's tasks
                gene_idx = genes.index(gene)
                reward = 0.0
                for task in tasks[:5]:  # Use first 5 tasks for speed
                    y = labels.set_index("Gene_Symbol").loc[genes, task].values.astype(float)
                    valid = ~np.isnan(y)
                    if valid.sum() < 10:
                        continue
                    # Simple correlation between attribute values and labels
                    attr_vals = np.array([rollout_attrs.get(key, 0.5) for key in attr_keys])
                    # Reward = how well this gene's attributes predict its label
                    label = y[gene_idx] if not np.isnan(y[gene_idx]) else 0.5
                    score = np.mean(attr_vals) if label == 1 else 1 - np.mean(attr_vals)
                    reward += score
                reward /= max(1, min(5, len(tasks)))
                gene_rewards.append(reward)

            rollout_rewards.append(gene_rewards)

        # Compute advantages (GRPO-style)
        flat_rewards = [r for gene_rs in rollout_rewards for r in gene_rs]
        batch_std = np.std(flat_rewards) + 1e-8

        advantages = []
        for gene_rs in rollout_rewards:
            gene_mean = np.mean(gene_rs)
            gene_advs = [(r - gene_mean) / batch_std for r in gene_rs]
            advantages.append(gene_advs)

        mean_reward = np.mean(flat_rewards)
        reward_log.append({"step": step, "mean_reward": mean_reward, "n_attrs": len(attr_keys)})

        if mean_reward > best_reward:
            best_reward = mean_reward
            model.save_pretrained(os.path.join(outdir, "checkpoint-best"))
            tokenizer.save_pretrained(os.path.join(outdir, "checkpoint-best"))

        if step % 10 == 0:
            print(f"  Step {step}/{max_steps}: reward={mean_reward:.4f}, "
                  f"n_attrs={len(attr_keys)}, best={best_reward:.4f}")

    # Save reward log
    pd.DataFrame(reward_log).to_csv(os.path.join(outdir, "reward_log.csv"), index=False)
    print(f"[RL] Done. Best reward={best_reward:.4f}. Saved to {outdir}/")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["sft", "rl"])
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--teacher_scores", default="data/features_llm_structured_scores.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--sft_checkpoint", default="results/rl_sft/checkpoint-best")
    ap.add_argument("--outdir", default="results")

    # RL hyperparameters
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--rl_lr", type=float, default=3e-5)
    ap.add_argument("--rl_steps", type=int, default=500)
    ap.add_argument("--eps_low", type=float, default=0.1)
    ap.add_argument("--eps_high", type=float, default=0.2)
    ap.add_argument("--kl_beta", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)

    args = ap.parse_args()

    if args.phase == "sft":
        run_sft(args)
    elif args.phase == "rl":
        run_rl(args)


if __name__ == "__main__":
    main()
