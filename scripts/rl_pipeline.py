#!/usr/bin/env python3
"""
RL Pipeline: GPT-4o-mini reasoning traces → SFT distillation → GRPO optimization

Step 1: Generate rich reasoning traces with GPT-4o-mini (open-ended attributes)
Step 2: SFT Qwen2.5-1.5B on these traces
Step 3: RL (GRPO) with Adjusted F1 reward on labeled subset
Step 4: Generate final attributes for all genes

Usage:
  python -u scripts/rl_pipeline.py --step traces   # Step 1: GPT-4o-mini traces
  python -u scripts/rl_pipeline.py --step sft       # Step 2: SFT
  python -u scripts/rl_pipeline.py --step rl        # Step 3: RL
  python -u scripts/rl_pipeline.py --step generate  # Step 4: Final generation
  python -u scripts/rl_pipeline.py --step all        # All steps
"""

import os, sys, json, re, argparse, time
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── Config ───────────────────────────────────────────────────────────────────
REASONING_PROMPT = """You are a senior drug discovery scientist analyzing a gene for therapeutic target identification.

Gene: {gene}
NCBI Summary: {summary}
Known label: {label_info}

Think step-by-step about this gene's therapeutic potential:
1. Analyze the gene's molecular function and biological role
2. Assess druggability, localization, pathway importance
3. Consider any properties relevant for target identification

Then output a JSON with your assessment. You may include ANY attributes you find relevant.
Each attribute should be a score in [0.0, 1.0].

Think carefully, then output ONLY a JSON object."""

SYS_PROMPT = ("You are a drug discovery scientist. Analyze the gene and output a JSON "
              "with therapeutic attribute scores in [0.0, 1.0]. Return STRICT JSON only.")

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

def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate reasoning traces with GPT-4o-mini
# ══════════════════════════════════════════════════════════════════════════════
def step_traces(args):
    """Generate open-ended reasoning traces for labeled gene subset."""
    from llm.llm_backends import AzureOpenAIBackend

    print("[TRACES] Loading data...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    labels = pd.read_csv(args.labels, sep="\t")
    tasks = [c for c in labels.columns if c.startswith("task_")]

    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))

    # Select subset: all positives + sampled negatives
    all_genes = set(labels["Gene_Symbol"].astype(str)) & set(g2s.keys())
    positives = set()
    for t in tasks:
        pos = labels[labels[t] == 1]["Gene_Symbol"].astype(str)
        positives.update(pos)
    positives = positives & all_genes

    negatives = all_genes - positives
    n_neg = min(len(negatives), len(positives) * 2)  # 2:1 ratio
    neg_sample = set(np.random.RandomState(42).choice(list(negatives), n_neg, replace=False))
    subset = sorted(positives | neg_sample)
    print(f"  Subset: {len(positives)} positives + {len(neg_sample)} negatives = {len(subset)}")

    # Check for existing traces (resume)
    outpath = os.path.join(args.outdir, "reasoning_traces.jsonl")
    existing = {}
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                obj = json.loads(line)
                existing[obj["gene"]] = obj
        print(f"  Resuming: {len(existing)} existing traces")

    todo = [g for g in subset if g not in existing]
    print(f"  Generating {len(todo)} new traces...")

    def _key(fname):
        path = os.path.expanduser(f"~/.secrets/{fname}")
        if os.path.isfile(path):
            return open(path).read().strip()
        raise FileNotFoundError(f"Key not found: {path}")

    backend = AzureOpenAIBackend(
        chat_endpoint="https://duanz-mkv4bpb9-eastus2.cognitiveservices.azure.com",
        chat_deployment="gpt-4o-mini",
        chat_api_version="2024-02-01",
        chat_api_key=_key("azure_openai_eastus2_key.txt"),
        embed_endpoint="https://duanz-mkv4bpb9-westus.cognitiveservices.azure.com",
        embed_deployment="text-embedding-3-large",
        embed_api_version="2024-02-01",
        embed_api_key=_key("azure_openai_westus_key.txt"),
    )

    with open(outpath, "a") as fout:
        for i, gene in enumerate(todo):
            summary = g2s[gene]
            # Determine label info
            gene_labels = {}
            for t in tasks:
                val = labels[labels["Gene_Symbol"] == gene][t].values
                if len(val) > 0 and not np.isnan(val[0]):
                    gene_labels[t] = int(val[0])
            label_info = "positive in: " + ", ".join(t for t, v in gene_labels.items() if v == 1) if any(v == 1 for v in gene_labels.values()) else "no known positive labels"

            prompt = REASONING_PROMPT.format(gene=gene, summary=summary, label_info=label_info)

            try:
                response = backend._call_llm(
                    system="You are a senior drug discovery scientist.",
                    user=prompt,
                )
                attrs = extract_attrs(response)
                record = {"gene": gene, "summary": summary, "response": response,
                         "attrs": attrs, "n_attrs": len(attrs)}
                fout.write(json.dumps(record) + "\n")
                fout.flush()

                if (i+1) % 100 == 0:
                    print(f"  {i+1}/{len(todo)} done")
            except Exception as e:
                print(f"  Error for {gene}: {e}")
                continue

    # Summary
    all_traces = []
    with open(outpath) as f:
        for line in f:
            all_traces.append(json.loads(line))

    all_keys = set()
    for t in all_traces:
        all_keys.update(t["attrs"].keys())
    print(f"\n[TRACES] Done: {len(all_traces)} traces, {len(all_keys)} unique attributes")
    print(f"  Top attributes: {sorted(all_keys)[:20]}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: SFT distillation
# ══════════════════════════════════════════════════════════════════════════════
def step_sft(args):
    """SFT Qwen2.5-1.5B on GPT-4o-mini reasoning traces."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("[SFT] Loading traces...")
    traces = []
    with open(os.path.join(args.outdir, "reasoning_traces.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            if obj["attrs"]:  # only valid traces
                traces.append(obj)
    print(f"  {len(traces)} valid traces")

    print(f"[SFT] Loading {args.model}...")
    device = "cuda"
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

    # Build training texts
    def make_prompt(gene, summary):
        msgs = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Gene: {gene}\nSummary: {summary}\nReturn JSON:"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    train_texts = []
    for t in traces:
        prompt = make_prompt(t["gene"], t["summary"])
        target = json.dumps(t["attrs"])
        train_texts.append(prompt + target + tokenizer.eos_token)

    print(f"[SFT] Training on {len(train_texts)} examples...")
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
            if n % 200 == 0:
                print(f"  Epoch {epoch+1}, batch {n}: loss={total_loss/n:.4f}")
        print(f"  [SFT] Epoch {epoch+1} done, loss={total_loss/n:.4f}")

    sft_path = os.path.join(args.outdir, "sft_checkpoint")
    model.save_pretrained(sft_path)
    tokenizer.save_pretrained(sft_path)
    print(f"  Saved to {sft_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: RL (GRPO) on labeled subset
# ══════════════════════════════════════════════════════════════════════════════
def step_rl(args):
    """GRPO RL optimization on labeled gene subset."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    device = "cuda"
    sft_path = os.path.join(args.outdir, "sft_checkpoint")

    print("[RL] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(sft_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, sft_path, is_trainable=True)

    # Load labeled subset
    print("[RL] Loading data...")
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    labels = pd.read_csv(args.labels, sep="\t")
    tasks = [c for c in labels.columns if c.startswith("task_")][:8]

    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))

    # Use trace genes as RL subset
    traces = []
    with open(os.path.join(args.outdir, "reasoning_traces.jsonl")) as f:
        for line in f:
            traces.append(json.loads(line))
    rl_genes = sorted(set(t["gene"] for t in traces) & set(g2s.keys()))
    print(f"  RL subset: {len(rl_genes)} genes")

    def make_prompt(gene, summary):
        msgs = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Gene: {gene}\nSummary: {summary}\nReturn JSON:"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Pre-generate baseline attrs for RL subset
    print("[RL] Generating baseline attributes...")
    model.eval()
    baseline = {}
    for i, gene in enumerate(rl_genes):
        prompt = make_prompt(gene, g2s[gene])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, temperature=0.1,
                               do_sample=True, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        baseline[gene] = extract_attrs(text)
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(rl_genes)}")

    # Compute baseline reward
    def compute_reward(attrs_dict, genes, labels_df, tasks):
        all_keys = sorted(set(k for v in attrs_dict.values() for k in v))
        if not all_keys:
            return 0.0
        mat = np.array([[attrs_dict.get(g, {}).get(k, 0.5) for k in all_keys] for g in genes])
        X = StandardScaler().fit_transform(mat)
        af1s = []
        for t in tasks:
            y = labels_df.set_index("Gene_Symbol").reindex(genes)[t].values.astype(float)
            valid = ~np.isnan(y)
            if valid.sum() < 20 or y[valid].sum() < 5:
                continue
            try:
                lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", solver="lbfgs")
                lr.fit(X[valid], y[valid].astype(int))
                prob = lr.predict_proba(X[valid])[:, 1]
                af1s.append(adjusted_f1(prob, y[valid].astype(int)))
            except:
                pass
        return float(np.mean(af1s)) if af1s else 0.0

    base_reward = compute_reward(baseline, rl_genes, labels, tasks)
    print(f"  Baseline Adjusted F1: {base_reward:.4f}")

    # RL loop
    G = args.group_size
    B = args.genes_per_step
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.rl_lr
    )

    log = []
    best_reward = base_reward

    print(f"\n[RL] GRPO: {args.rl_steps} steps, G={G}, B={B}")
    for step in range(args.rl_steps):
        t0 = time.time()
        batch = list(np.random.choice(rl_genes, size=min(B, len(rl_genes)), replace=False))

        # Generate rollouts
        model.eval()
        rollouts = []
        for gene in batch:
            prompt = make_prompt(gene, g2s[gene])
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
            for _ in range(G):
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=200, temperature=0.8,
                                       top_k=20, do_sample=True, pad_token_id=tokenizer.pad_token_id)
                gen_ids = out[0][enc["input_ids"].shape[1]:]
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                attrs = extract_attrs(text)
                rollouts.append((gene, attrs, enc["input_ids"][0], gen_ids))

        # Compute rewards
        rewards = []
        for gene, attrs, _, _ in rollouts:
            modified = {g: baseline[g] for g in rl_genes}
            modified[gene] = attrs if attrs else baseline.get(gene, {})
            r = compute_reward(modified, rl_genes, labels, tasks)
            rewards.append(r)
        rewards = np.array(rewards)

        # Advantages
        batch_std = rewards.std() + 1e-8
        advantages = []
        idx = 0
        for gene in batch:
            gene_rs = rewards[idx:idx+G]
            gene_mean = gene_rs.mean()
            for r in gene_rs:
                advantages.append((r - gene_mean) / batch_std)
            idx += G
        advantages = np.array(advantages)

        # Policy update
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_valid = 0
        for i, (gene, attrs, inp_ids, gen_ids) in enumerate(rollouts):
            adv = advantages[i]
            if abs(adv) < 1e-8 or len(gen_ids) == 0:
                continue
            full_ids = torch.cat([inp_ids, gen_ids]).unsqueeze(0).to(device)
            lab = full_ids.clone()
            lab[0, :len(inp_ids)] = -100
            out = model(input_ids=full_ids, labels=lab)
            pg_loss = out.loss * (-adv)
            pg_loss.backward()
            total_loss += pg_loss.item()
            n_valid += 1

        if n_valid > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= n_valid
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update baseline with best rollouts
        idx = 0
        for gene in batch:
            gene_rs = rewards[idx:idx+G]
            best_j = gene_rs.argmax()
            best_attrs = rollouts[idx + best_j][1]
            if best_attrs:
                baseline[gene] = best_attrs
            idx += G

        mean_r = rewards.mean()
        n_attrs = len(set(k for _, attrs, _, _ in rollouts for k in attrs))
        dt = time.time() - t0

        log.append({"step": step, "mean_reward": float(mean_r),
                    "best_step_reward": float(rewards.max()), "n_attrs": n_attrs,
                    "pg_loss": float(total_loss/max(n_valid,1)), "time_s": dt})

        if mean_r > best_reward:
            best_reward = mean_r
            model.save_pretrained(os.path.join(args.outdir, "rl_best"))
            tokenizer.save_pretrained(os.path.join(args.outdir, "rl_best"))

        if step % 5 == 0:
            print(f"  Step {step:3d}: reward={mean_r:.4f}±{rewards.std():.4f}, "
                  f"attrs={n_attrs}, loss={total_loss/max(n_valid,1):.4f}, {dt:.0f}s")

    pd.DataFrame(log).to_csv(os.path.join(args.outdir, "rl_log.csv"), index=False)
    print(f"\n[RL] Done. Baseline: {base_reward:.4f} → Best: {best_reward:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Generate final attributes for all genes
# ══════════════════════════════════════════════════════════════════════════════
def step_generate(args):
    """Generate RL-optimized attributes for all genes."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda"
    rl_path = os.path.join(args.outdir, "rl_best")
    if not os.path.exists(rl_path):
        rl_path = os.path.join(args.outdir, "sft_checkpoint")

    print(f"[GEN] Loading model from {rl_path}...")
    tokenizer = AutoTokenizer.from_pretrained(rl_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, rl_path)
    model.eval()

    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    genes = sorted(g2s.keys())

    def make_prompt(gene, summary):
        msgs = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Gene: {gene}\nSummary: {summary}\nReturn JSON:"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f"[GEN] Generating attributes for {len(genes)} genes...")
    all_attrs = {}
    for i, gene in enumerate(genes):
        prompt = make_prompt(gene, g2s[gene])
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, temperature=0.1,
                               do_sample=True, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        all_attrs[gene] = extract_attrs(text)
        if (i+1) % 1000 == 0:
            valid = sum(1 for v in all_attrs.values() if v)
            print(f"  {i+1}/{len(genes)}, {valid} valid")

    keys = sorted(set(k for v in all_attrs.values() for k in v))
    rows = [{"Gene_Symbol": g, **{k: all_attrs[g].get(k, np.nan) for k in keys}} for g in genes]
    out_csv = os.path.join(args.outdir, "features_rl_attributes.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[DONE] {len(keys)} attributes, saved to {out_csv}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["traces", "sft", "rl", "generate", "all"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results/rl_pipeline")
    ap.add_argument("--sft_epochs", type=int, default=3)
    ap.add_argument("--rl_steps", type=int, default=200)
    ap.add_argument("--rl_lr", type=float, default=5e-5)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--genes_per_step", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.step in ("traces", "all"):
        step_traces(args)
    if args.step in ("sft", "all"):
        step_sft(args)
    if args.step in ("rl", "all"):
        step_rl(args)
    if args.step in ("generate", "all"):
        step_generate(args)

if __name__ == "__main__":
    main()
