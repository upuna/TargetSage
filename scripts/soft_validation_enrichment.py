#!/usr/bin/env python3
"""
Soft validation: train on 2025 labels, rank all genes, then check if
top-ranked unlabeled genes are enriched in ChEMBL Phase 2/3 clinical targets.

Usage:
    python scripts/soft_validation_enrichment.py \
        --ranking  results/inference_2025_tclin/pharos_tclin_vs_others_ranking.csv \
        --labels   data/gene_labels.tsv \
        --ct_genes data/chembl_clinical_targets_phase23.csv \
        --task     task_pharos_tclin_vs_others \
        --out      figures/soft_validation.pdf
"""

import argparse, glob
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})

METHOD_COLORS = {
    "TargetSage": "#5B4A8B",
    "GB":         "#C4B08A",
    "RF":         "#9BAE93",
    "LR":         "#8FA5B5",
    "SVM":        "#BC9A8E",
    "MLP":        "#B5ADA0",
    "KNN":        "#C6B684",
    "NB":         "#A8AD93",
    "ResNet":     "#A4B8C4",
}


def get_unlabeled(ranking_csv, positives, ct_genes, tdl_filter=None, label_df=None):
    df = pd.read_csv(ranking_csv).sort_values("rank").reset_index(drop=True)
    score_col = [c for c in df.columns if "score" in c.lower()][0]
    df = df.rename(columns={score_col: "score"})
    df["is_positive"] = df["Gene_Symbol"].isin(positives)
    df["is_ct"]       = df["Gene_Symbol"].str.upper().isin(ct_genes)
    if tdl_filter is not None and label_df is not None:
        tdl_map = dict(zip(label_df["Gene_Symbol"], label_df["idgTDL_new"].str.upper()))
        df["tdl"] = df["Gene_Symbol"].map(tdl_map)
        df = df[df["tdl"].isin(tdl_filter)]
    return df[~df["is_positive"]].copy().reset_index(drop=True)


def enrichment_curve(unlabeled, step=1):
    M = len(unlabeled)
    K = unlabeled["is_ct"].sum()
    base = K / M
    pcts, folds = [], []
    for pct in range(step, 101, step):
        n = max(1, int(M * pct / 100))
        frac = unlabeled.iloc[:n]["is_ct"].mean()
        pcts.append(pct)
        folds.append(frac / base)
    return np.array(pcts), np.array(folds)


def odds_ratio_at_topk(unlabeled, pct):
    M = len(unlabeled)
    K = unlabeled["is_ct"].sum()
    n_top = max(1, int(M * pct / 100))
    n_ct_top  = unlabeled.iloc[:n_top]["is_ct"].sum()
    n_ct_rest = K - n_ct_top
    n_rest    = M - n_top
    table = [[n_ct_top, n_top - n_ct_top],
             [n_ct_rest, n_rest - n_ct_rest]]
    or_, pval = fisher_exact(table, alternative="greater")
    return or_, pval, int(n_ct_top), n_top


def plot_comparison(methods_data, out):
    """
    methods_data: dict  method_name -> (curve_pcts, curve_folds, or5pct)
    """
    methods = list(methods_data.keys())
    # sort by OR at 5%
    methods = sorted(methods, key=lambda m: methods_data[m][2], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: enrichment curves ────────────────────────────────────────────
    ax = axes[0]
    for method in methods:
        pcts, folds, _, _ = methods_data[method]
        color = METHOD_COLORS.get(method, "#AAAAAA")
        lw = 2.5 if method == "TargetSage" else 1.2
        ls = "-"  if method == "TargetSage" else "--"
        alpha = 1.0 if method == "TargetSage" else 0.7
        ax.plot(pcts, folds, color=color, linewidth=lw,
                linestyle=ls, alpha=alpha, label=method)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="Background")
    ax.axvline(1.0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_xlim(0, 20)
    ax.set_xlabel("Top-k% of unlabeled Tbio genes", fontsize=10)
    ax.set_ylabel("Fold enrichment over background", fontsize=10)
    ax.set_title("ChEMBL Phase 2/3 Enrichment (Tbio genes, top 20%)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: OR at Top 5% bar chart ──────────────────────────────────────
    ax2 = axes[1]
    ors   = [methods_data[m][2] for m in methods]
    pvals = [methods_data[m][3] for m in methods]
    colors = [METHOD_COLORS.get(m, "#AAAAAA") for m in methods]
    x = np.arange(len(methods))
    bars = ax2.bar(x, ors, color=colors, alpha=0.88, width=0.6,
                   edgecolor="white", linewidth=0.5)

    for i, (or_, pval) in enumerate(zip(ors, pvals)):
        stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax2.text(i, or_ + 0.1, stars, ha="center", va="bottom",
                 fontsize=8, color=colors[i], fontweight="bold")

    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9, rotation=30, ha="right")
    ax2.set_ylabel("Odds Ratio (Top 1%)", fontsize=10)
    ax2.set_title("Method Comparison at Top 1% Cutoff", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Soft Validation: ChEMBL Phase 2/3 Enrichment among PHAROS Tbio Genes\n"
        "(trained on 2025 labels; Tbio = understudied proteins, n=11,891)",
        fontsize=10, fontweight="bold", y=1.02
    )

    plt.savefig(out, dpi=300, bbox_inches="tight")
    svg = out.rsplit(".", 1)[0] + ".svg"
    plt.savefig(svg, format="svg", bbox_inches="tight")
    print(f"[DONE] {out}\n[DONE] {svg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targetsage_ranking", required=True,
                    help="TargetSage ranking CSV")
    ap.add_argument("--baseline_dir",       required=True,
                    help="Dir with *_ranking.csv for baselines")
    ap.add_argument("--labels",     default="data/gene_labels.tsv")
    ap.add_argument("--ct_genes",   default="data/chembl_clinical_targets_phase23.csv")
    ap.add_argument("--task",       default="task_pharos_tclin_vs_others")
    ap.add_argument("--topk_bar",   type=float, default=1.0,
                    help="Top-k%% used for the bar chart comparison")
    ap.add_argument("--tdl_filter", default="TBIO",
                    help="Comma-separated PHAROS TDL categories to restrict analysis (e.g. TBIO,TDARK)")
    ap.add_argument("--out",        default="figures/soft_validation_comparison.pdf")
    args = ap.parse_args()

    # load labels and CT genes
    label_df = pd.read_csv(args.labels, sep="\t")
    label_df[args.task] = pd.to_numeric(label_df[args.task], errors="coerce").fillna(0).astype(int)
    positives = set(label_df[label_df[args.task] == 1]["Gene_Symbol"].astype(str))
    ct_df     = pd.read_csv(args.ct_genes)
    ct_genes  = set(ct_df["gene_symbol"].astype(str).str.upper())

    tdl_filter = set(t.strip().upper() for t in args.tdl_filter.split(",")) if args.tdl_filter else None

    ct_unlabeled = len(ct_genes - positives)
    print("Training positives: %d" % len(positives))
    print("CT Phase2/3 genes:  %d  (%d independent)" % (len(ct_genes), ct_unlabeled))
    if tdl_filter:
        print("TDL filter: %s" % tdl_filter)

    methods_data = {}

    # TargetSage
    ul = get_unlabeled(args.targetsage_ranking, positives, ct_genes, tdl_filter, label_df)
    pcts, folds = enrichment_curve(ul)
    or_, pval, _, _ = odds_ratio_at_topk(ul, args.topk_bar)
    methods_data["TargetSage"] = (pcts, folds, or_, pval)
    print("TargetSage  OR@%.0f%%=%.2f  p=%.2e  (n=%d, ct=%d)" % (
        args.topk_bar, or_, pval, len(ul), ul["is_ct"].sum()))

    # Baselines
    for csv in sorted(glob.glob("%s/*_ranking.csv" % args.baseline_dir)):
        name = os.path.basename(csv).replace("_ranking.csv", "")
        ul = get_unlabeled(csv, positives, ct_genes, tdl_filter, label_df)
        pcts, folds = enrichment_curve(ul)
        or_, pval, _, _ = odds_ratio_at_topk(ul, args.topk_bar)
        methods_data[name] = (pcts, folds, or_, pval)
        print("%-12s OR@%.0f%%=%.2f  p=%.2e" % (name, args.topk_bar, or_, pval))

    plot_comparison(methods_data, args.out)


if __name__ == "__main__":
    import os
    main()
