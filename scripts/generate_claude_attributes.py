#!/usr/bin/env python3
"""
Generate LLM attribute scores using Claude API.
Produces features_claude_structured_scores.csv comparable to GPT-4o-mini version.
"""

import os, sys, json, time, re
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ATTR_KEYS = [
    "druggability_score", "secreted_probability", "membrane_localization",
    "enzyme_activity", "kinase_family", "gpcr_family",
    "ion_channel", "oncogenic_driver", "tumor_suppressor",
    "pathway_centrality", "clinical_evidence", "safety_concern",
]

SYSTEM_PROMPT = """You are a senior drug discovery scientist.

Your task is to extract calibrated, continuous-valued therapeutic attributes
for target identification.

Calibration rules:
- Values must lie in [0.0, 1.0]; avoid exact 0.0 or 1.0 unless the property
  is a strict binary definition (e.g. kinase vs non-kinase).
- For well-established targets prefer high but non-saturating values
  (e.g. 0.85-0.95 instead of 1.0).
- Safety concern reflects known class-effect or mechanism-based toxicities,
  not drug approval success.
- Use the full dynamic range; values will be compared across ~19k genes.

Return STRICT JSON only (no markdown, no extra text)."""

USER_TEMPLATE = """Analyze this gene for therapeutic target identification:

Gene Symbol: {gene_symbol}
Gene Summary: {gene_summary}

Estimate these attributes. ALL values must be in [0.0, 1.0] and rounded to
EXACTLY 2 decimal places (e.g. 0.85, 0.03, 0.50).

1. druggability_score: Overall druggability potential
2. secreted_probability: Secreted protein likelihood
3. membrane_localization: Membrane localization
4. enzyme_activity: Enzyme activity
5. kinase_family: Kinase membership
6. gpcr_family: GPCR membership
7. ion_channel: Ion channel
8. oncogenic_driver: Oncogene potential
9. tumor_suppressor: TSG potential
10. pathway_centrality: Pathway importance
11. clinical_evidence: Clinical evidence strength
12. safety_concern: Known safety risk

Return ONLY this JSON (no other text):
{{"druggability_score": 0.XX, "secreted_probability": 0.XX,
"membrane_localization": 0.XX, "enzyme_activity": 0.XX,
"kinase_family": 0.XX, "gpcr_family": 0.XX, "ion_channel": 0.XX,
"oncogenic_driver": 0.XX, "tumor_suppressor": 0.XX,
"pathway_centrality": 0.XX, "clinical_evidence": 0.XX,
"safety_concern": 0.XX}}"""


def call_claude(client, gene, summary):
    """Call Claude API for one gene."""
    user_msg = USER_TEMPLATE.format(gene_symbol=gene, gene_summary=summary)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text
        # Extract JSON
        m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            row = {"Gene_Symbol": gene}
            for k in ATTR_KEYS:
                val = obj.get(k, None)
                if val is not None:
                    row[k] = round(float(val), 2)
            return row
    except Exception as e:
        print(f"  Error for {gene}: {e}")
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--output", default="data/features_claude_structured_scores.csv")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Load API key
    key_path = os.path.expanduser("~/.secrets/anthropic_api_key.txt")
    api_key = open(key_path).read().strip()

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    # Load gene summaries
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    genes = sorted(g2s.keys())
    print(f"Total genes: {len(genes)}")

    # Resume support
    existing = {}
    if args.resume and os.path.exists(args.output):
        df = pd.read_csv(args.output)
        existing = set(df["Gene_Symbol"].astype(str))
        print(f"Resuming: {len(existing)} already done")

    todo = [g for g in genes if g not in existing]
    print(f"To generate: {len(todo)}")

    # Process with thread pool
    results = []
    if args.resume and os.path.exists(args.output):
        results = pd.read_csv(args.output).to_dict("records")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(call_claude, client, g, g2s[g]): g for g in todo}
        for future in as_completed(futures):
            row = future.result()
            if row:
                results.append(row)
            done += 1
            if done % 100 == 0:
                # Save checkpoint
                pd.DataFrame(results).to_csv(args.output, index=False)
                print(f"  {done}/{len(todo)} done, {len(results)} valid, saved")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n[DONE] {len(df)} genes saved to {args.output}")


if __name__ == "__main__":
    main()
