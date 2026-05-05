#!/usr/bin/env python3
"""GPT-as-classifier baseline: directly prompt GPT-4o-mini to score each gene's
therapeutic potential for each task, without any learned classifier on top.

For each gene + task, prompt GPT-4o-mini with gene summary + task description,
ask for a 0-1 druggability score, and evaluate via Adjusted F1.
"""
import os, sys, json, argparse, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT = "/home/zihend1/Genesis/TargetSage2"
os.chdir(ROOT)
sys.path.insert(0, "scripts")

# Azure OpenAI setup
try:
    from openai import AzureOpenAI
    AZURE_KEY = open(os.path.expanduser("~/.secrets/azure_openai_eastus2_key.txt")).read().strip()
    client = AzureOpenAI(
        azure_endpoint="https://duanz-mkv4bpb9-eastus2.cognitiveservices.azure.com",
        api_key=AZURE_KEY,
        api_version="2024-02-01",
    )
    DEPLOYMENT = "gpt-4o-mini"
except Exception as e:
    print(f"OpenAI setup failed: {e}")
    sys.exit(1)


TASK_PROMPT = {
    "task_pharos_tclin_vs_others": "Is this gene a clinically approved drug target (PHAROS Tclin)? Score 0-1 based on evidence of clinical-grade drug targeting.",
    "task_pharos_tclin_tchem_vs_others": "Is this gene a clinical or chemical drug target (PHAROS Tclin/Tchem)? Score 0-1.",
    "task_triage_tier1_vs_others": "Is this gene a top-tier drug target (tractability tier 1)? Score 0-1.",
    "task_triage_tier12_vs_others": "Is this gene a high-confidence drug target (tier 1-2)? Score 0-1.",
    "task_cancer_druggability": "Is this gene a druggable cancer target? Score 0-1.",
    "task_cancer_type_specific_target_prioritization": "Is this gene a cancer type-specific target? Score 0-1.",
    "task_pan_cancer_target_prioritization": "Is this gene a pan-cancer therapeutic target? Score 0-1.",
    "task_T1_targets_only": "Is this gene a Tier-1 cancer target with validated drug interactions? Score 0-1.",
    "task_T1_T2_targets": "Is this gene a Tier 1-2 cancer target? Score 0-1.",
    "task_T1_T2_T3_targets": "Is this gene a Tier 1-3 cancer target? Score 0-1.",
    "task_sm_bucket1_vs_others": "Is this gene targeted by an approved small molecule drug? Score 0-1.",
    "task_sm_bucket123_vs_others": "Is this gene targeted by approved or clinical small molecules? Score 0-1.",
    "task_ab_bucket1_vs_others": "Is this gene targeted by an approved antibody drug? Score 0-1.",
    "task_ab_bucket123_vs_others": "Is this gene targeted by approved or clinical antibodies? Score 0-1.",
    "task_protac_bucket1234_vs_others": "Is this gene susceptible to PROTAC-mediated degradation? Score 0-1.",
}


def query_gene(gene, summary, task_prompt, max_retries=2):
    """Query GPT once for a single gene."""
    msg = [
        {"role": "system", "content": "You are a drug discovery expert. Given a gene and its summary, score the likelihood that it meets the specified therapeutic criterion. Respond with a single JSON object {\"score\": <float in [0,1]>} and nothing else."},
        {"role": "user", "content": f"Gene: {gene}\nSummary: {summary[:400]}\n\nQuestion: {task_prompt}\n\nScore 0-1:"},
    ]
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT, messages=msg,
                temperature=0.0, max_tokens=50, timeout=15,
            )
            text = resp.choices[0].message.content.strip()
            m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            if m:
                return min(max(float(m.group(1)), 0.0), 1.0)
            # fallback: any number
            m = re.search(r'([0-9]*\.[0-9]+)', text)
            if m:
                v = float(m.group(1))
                if 0 <= v <= 1:
                    return v
            return 0.5
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            return 0.5


def adjusted_f1(probs, y):
    pos = y == 1
    if pos.sum() == 0:
        return 0.0
    R_soft = probs[pos].mean()
    p_bar = probs.mean()
    return float(R_soft ** 2 / max(p_bar, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_subset", type=int, default=500, help="Subset size per task")
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--outdir", default="results/gpt_classifier")
    ap.add_argument("--tasks", default="", help="Comma-separated tasks; empty=all")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[DATA] Loading...")
    summaries = pd.read_csv("data/gene_summaries.tsv", sep="\t")
    labels_df = pd.read_csv("data/gene_labels.tsv", sep="\t")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))

    task_list = args.tasks.split(",") if args.tasks else list(TASK_PROMPT.keys())
    task_list = [t.strip() for t in task_list if t.strip()]

    all_results = []
    for task in task_list:
        print(f"\n=== {task} ===")
        task_prompt = TASK_PROMPT[task]
        y_all = labels_df.set_index("Gene_Symbol")[task]

        pos = y_all[y_all == 1].index.tolist()
        neg = y_all[y_all == 0].index.tolist()

        # Sample subset (same as other methods, ~500 genes)
        np.random.seed(42)
        target_pos = min(len(pos), max(int(args.n_subset * 0.6), 100))
        target_neg = min(len(neg), args.n_subset - target_pos)
        pos_s = np.random.choice(pos, target_pos, replace=False).tolist() if target_pos < len(pos) else pos
        neg_s = np.random.choice(neg, target_neg, replace=False).tolist()
        subset = pos_s + neg_s
        subset = [g for g in subset if g in g2s]
        y_subset = np.array([1 if g in pos else 0 for g in subset])
        print(f"  Subset: {len(subset)} genes ({y_subset.sum()} pos, {len(y_subset)-y_subset.sum()} neg)")

        # Parallel query
        t0 = time.time()
        scores_map = {}
        with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
            futures = {ex.submit(query_gene, g, g2s.get(g, ""), task_prompt): g for g in subset}
            done = 0
            for fut in as_completed(futures):
                g = futures[fut]
                scores_map[g] = fut.result()
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(subset)}... ({time.time()-t0:.0f}s)")

        scores = np.array([scores_map[g] for g in subset])
        af1 = adjusted_f1(scores, y_subset) * 100
        print(f"  GPT-classifier Adj F1: {af1:.2f}%  ({time.time()-t0:.0f}s)")

        # Save
        pd.DataFrame({"gene": subset, "y": y_subset, "gpt_score": scores}).to_csv(
            f"{args.outdir}/{task}.csv", index=False
        )
        all_results.append({"task": task, "n": len(subset), "n_pos": int(y_subset.sum()), "adj_f1": af1})

        pd.DataFrame(all_results).to_csv(f"{args.outdir}/summary.csv", index=False)

    print("\n=== Summary ===")
    df = pd.DataFrame(all_results)
    for _, r in df.iterrows():
        print(f"  {r['task']}: {r['adj_f1']:.2f}% (n={r['n']}, pos={r['n_pos']})")
    print(f"  MEAN: {df['adj_f1'].mean():.2f}%")


if __name__ == "__main__":
    main()
