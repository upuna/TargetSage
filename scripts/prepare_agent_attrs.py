#!/usr/bin/env python3
"""
Prepare agent_attributes.csv for training:
  1. Filter columns to those with >= min_coverage non-NaN fraction
  2. Report coverage statistics
  3. Save filtered CSV
"""
import argparse
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="results/agent_tools/agent_attributes.csv")
    ap.add_argument("--output", default="results/agent_tools/agent_attributes_filtered.csv")
    ap.add_argument("--min_coverage", type=float, default=0.05,
                    help="Min fraction of genes that must have a value (default 5%%)")
    ap.add_argument("--max_cols", type=int, default=200,
                    help="Max number of attribute columns to keep (default 200)")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  Shape: {df.shape[0]} genes x {df.shape[1]} columns (incl. Gene_Symbol)")

    # Coverage per column
    attr_cols = [c for c in df.columns if c != "Gene_Symbol"]
    n = len(df)
    coverage = {c: df[c].notna().sum() / n for c in attr_cols}

    # Sort by coverage descending
    sorted_cols = sorted(coverage.items(), key=lambda x: -x[1])

    print(f"\nTop 30 attributes by coverage:")
    for col, cov in sorted_cols[:30]:
        print(f"  {col:<60s} {cov*100:.1f}%")

    # Filter
    keep_cols = [c for c, cov in sorted_cols if cov >= args.min_coverage]
    print(f"\nColumns with >= {args.min_coverage*100:.0f}% coverage: {len(keep_cols)}")

    if len(keep_cols) > args.max_cols:
        keep_cols = keep_cols[:args.max_cols]
        print(f"Truncated to top {args.max_cols} columns")

    out_df = df[["Gene_Symbol"] + keep_cols].copy()
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved {out_df.shape[1]-1} attribute columns to {args.output}")

    # Final coverage stats
    final_coverage = out_df[keep_cols].notna().sum(axis=1)
    print(f"\nPer-gene attribute coverage after filtering:")
    print(f"  Mean attrs per gene: {final_coverage.mean():.1f}")
    print(f"  Median:              {final_coverage.median():.1f}")
    print(f"  Genes with >=5 attrs: {(final_coverage >= 5).sum()} ({(final_coverage >= 5).mean()*100:.1f}%)")
    print(f"  Genes with 0 attrs:   {(final_coverage == 0).sum()} ({(final_coverage == 0).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
