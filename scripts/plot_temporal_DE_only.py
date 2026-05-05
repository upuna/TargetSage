#!/usr/bin/env python3
"""Generate temporal validation D-E strip plots only (no ABC panels)."""

import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.linewidth": 0.8,
})

METHOD_COLORS = {
    "TargetSage": "#6E5C7A", "GB": "#C4B08A", "RF": "#9BAE93",
    "LR": "#8FA5B5", "SVM": "#BC9A8E", "MLP": "#B5ADA0",
    "KNN": "#C6B684", "NB": "#A8AD93", "TabNet": "#8AAAA5",
    "FT-Trans": "#C4A4A4", "ResNet": "#A4B8C4",
}
METHOD_LABELS = {
    "TargetSage_supervised_logits": "TargetSage",
    "TargetSage_pu_logits": "TargetSage", "TargetSage": "TargetSage",
    "GB": "GB", "RF_100": "RF", "RF": "RF", "LR": "LR",
    "SVM_RBF": "SVM", "SVM": "SVM", "MLP_128_64": "MLP", "MLP": "MLP",
    "KNN_k10": "KNN", "KNN": "KNN", "NB": "NB",
    "TabNet": "TabNet", "FT-Trans": "FT-Trans", "ResNet": "ResNet",
}

def get_label(raw): return METHOD_LABELS.get(raw, raw)

def load_task_data(results_dir, task_key):
    upgraded_csv = os.path.join(results_dir, f"{task_key}__upgraded_genes.csv")
    if not os.path.exists(upgraded_csv): return {}
    upgraded = pd.read_csv(upgraded_csv)["Gene_Symbol"].astype(str).tolist()
    data = {}
    for csv in sorted(glob.glob(os.path.join(results_dir, f"{task_key}__*.csv"))):
        fname = os.path.basename(csv)
        if fname.endswith("upgraded_genes.csv"): continue
        raw = fname.replace(f"{task_key}__", "").replace(".csv", "")
        label = get_label(raw)
        df = pd.read_csv(csv, index_col=0)
        N = len(df)
        g2r = dict(zip(df["Gene_Symbol"].astype(str), df.index.astype(int)))
        pcts = np.array([g2r[str(g)] / N * 100.0 for g in upgraded if str(g) in g2r])
        if len(pcts) and (label not in data or np.median(pcts) < np.median(data[label])):
            data[label] = pcts
    return data, len(upgraded)

def strip_plot(ax, data_dict, n_total, title="", ylabel=True):
    sorted_methods = sorted(data_dict, key=lambda m: np.median(data_dict[m]))
    rng = np.random.default_rng(0)
    for xi, method in enumerate(sorted_methods):
        pcts = data_dict[method]
        color = METHOD_COLORS.get(method, "#AAAAAA")
        jitter = rng.uniform(-0.25, 0.25, size=len(pcts))
        ax.scatter(xi + jitter, pcts, s=12, alpha=0.45, color=color, linewidths=0, zorder=3)
        med = np.median(pcts)
        q25, q75 = np.percentile(pcts, 25), np.percentile(pcts, 75)
        ax.add_patch(mpatches.FancyBboxPatch(
            (xi - 0.30, q25), 0.60, q75 - q25,
            boxstyle="square,pad=0", linewidth=0, facecolor=color, alpha=0.20, zorder=2))
        ax.plot([xi - 0.30, xi + 0.30], [med, med], color=color, linewidth=2.0, zorder=4)

    for xi, method in enumerate(sorted_methods):
        med = np.median(data_dict[method])
        color = METHOD_COLORS.get(method, "#AAAAAA")
        is_best = (xi == 0)
        ax.text(xi, -5.5, f"{med:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if is_best else "normal",
                color=color, clip_on=False)

    ax.set_xticks(range(len(sorted_methods)))
    ax.set_xticklabels(sorted_methods, fontsize=10)
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.6, len(sorted_methods) - 0.4)
    ax.set_ylim(88, -10)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(axis="y", labelsize=9)
    if ylabel:
        ax.set_ylabel("Rank Percentile (lower is better)", fontsize=10)
    ax.axhline(10, color="#AAAAAA", linestyle="--", linewidth=0.8, zorder=1)
    ax.text(len(sorted_methods) - 0.4, 10, "Top 10%", va="center", ha="left",
            fontsize=8, color="#AAAAAA")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out", default="69d801ea1fd28f1005350c54/figures/temporal_DE.pdf")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data_A, n_A = load_task_data(args.results_dir, "upgrade_to_Tclin")
    data_B, n_B = load_task_data(args.results_dir, "upgrade_to_TclinOrTchem")

    fig, (ax_D, ax_E) = plt.subplots(2, 1, figsize=(11, 4.2))
    fig.subplots_adjust(hspace=0.70)

    if data_A:
        strip_plot(ax_D, data_A, n_A,
                   title=f"Task A: Upgrade to Tclin (N={n_A} upgraded genes)")
    if data_B:
        strip_plot(ax_E, data_B, n_B,
                   title=f"Task B: Upgrade to Tclin/Tchem (N={n_B} upgraded genes)")

    for ax, label in [(ax_D, "A"), (ax_E, "B")]:
        ax.text(-0.05, 1.12, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    svg_out = args.out.rsplit(".", 1)[0] + ".svg"
    plt.savefig(svg_out, format="svg", bbox_inches="tight")
    print(f"[DONE] {args.out}")

if __name__ == "__main__":
    main()
