#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot temporal validation figure (Fig 3 style).
Outputs editable PDF + SVG with Arial for Illustrator.
"""

import os, argparse, glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Illustrator-friendly settings ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size":         10,
    "pdf.fonttype":      42,   # TrueType → editable in Illustrator
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# ── colour palette ────────────────────────────────────────────────────────────
# Morandi palette: low-saturation, desaturated earthy tones
TDL_COLORS = {
    "Tclin": "#5B4A8B", "Tchem": "#8B6BB1",
    "Tbio":  "#B8A0D4", "Tdark": "#DDD0EF",
}
METHOD_COLORS = {
    "GeneTrace": "#6E5C7A",
    "GB":        "#C4B08A",
    "RF":        "#9BAE93",
    "LR":        "#8FA5B5",
    "SVM":       "#BC9A8E",
    "MLP":       "#B5ADA0",
    "KNN":       "#C6B684",
    "NB":        "#A8AD93",
    "TabNet":    "#8AAAA5",
    "FT-Trans":  "#C4A4A4",
    "ResNet":    "#A4B8C4",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def normalize_tdl(x):
    if pd.isna(x): return np.nan
    return {"tclin":"Tclin","tchem":"Tchem","tbio":"Tbio","tdark":"Tdark"}.get(
        str(x).strip().lower(), str(x).strip())

METHOD_LABELS = {
    "TargetSage_supervised_logits": "GeneTrace",
    "TargetSage_supervised_calibrated_prob": "GeneTrace",
    "TargetSage_pu_logits": "GeneTrace",
    "TargetSage_pu_calibrated_prob": "GeneTrace",
    "TargetSage": "GeneTrace",
    "GeneTrace": "GeneTrace",
    "GB": "GB", "RF_100": "RF", "RF": "RF", "LR": "LR",
    "SVM_RBF": "SVM", "SVM": "SVM",
    "MLP_128_64": "MLP", "MLP": "MLP",
    "KNN_k10": "KNN", "KNN": "KNN", "NB": "NB",
    "TabNet": "TabNet", "FT-Trans": "FT-Trans",
}

def get_method_label(raw):
    return METHOD_LABELS.get(raw, raw)

def load_ranking_percentiles(ranking_csv, upgraded_genes):
    df = pd.read_csv(ranking_csv, index_col=0)
    N = len(df)
    g2r = dict(zip(df["Gene_Symbol"].astype(str), df.index.astype(int)))
    return np.array([g2r[str(g)] / N * 100.0 for g in upgraded_genes if str(g) in g2r])

def load_task_data(results_dir, task_key):
    upgraded_csv = os.path.join(results_dir, f"{task_key}__upgraded_genes.csv")
    if not os.path.exists(upgraded_csv): return {}
    upgraded = pd.read_csv(upgraded_csv)["Gene_Symbol"].astype(str).tolist()
    data = {}
    for csv in sorted(glob.glob(os.path.join(results_dir, f"{task_key}__*.csv"))):
        fname = os.path.basename(csv)
        if fname.endswith("upgraded_genes.csv"): continue
        raw = fname.replace(f"{task_key}__", "").replace(".csv", "")
        label = get_method_label(raw)
        pcts = load_ranking_percentiles(csv, upgraded)
        if len(pcts):
            if label not in data or np.median(pcts) < np.median(data[label]):
                data[label] = pcts
    return data


# ── strip plot ────────────────────────────────────────────────────────────────
def strip_plot(ax, data_dict, title="", ylabel=True):
    sorted_methods = sorted(data_dict, key=lambda m: np.median(data_dict[m]))
    rng = np.random.default_rng(0)

    for xi, method in enumerate(sorted_methods):
        pcts = data_dict[method]
        color = METHOD_COLORS.get(method, "#AAAAAA")
        jitter = rng.uniform(-0.25, 0.25, size=len(pcts))
        ax.scatter(xi + jitter, pcts, s=12, alpha=0.45, color=color,
                   linewidths=0, zorder=3)
        med = np.median(pcts)
        q25, q75 = np.percentile(pcts, 25), np.percentile(pcts, 75)
        ax.add_patch(mpatches.FancyBboxPatch(
            (xi - 0.30, q25), 0.60, q75 - q25,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=color, alpha=0.20, zorder=2))
        ax.plot([xi - 0.30, xi + 0.30], [med, med],
                color=color, linewidth=2.0, zorder=4)

    # median % above each column
    for xi, method in enumerate(sorted_methods):
        med = np.median(data_dict[method])
        color = METHOD_COLORS.get(method, "#AAAAAA")
        is_best = (xi == 0)
        ax.text(xi, -5.5, f"{med:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if is_best else "normal",
                color=color, clip_on=False)

    ax.set_xticks(range(len(sorted_methods)))
    ax.set_xticklabels(sorted_methods, fontsize=11, fontweight="normal")
    ax.get_xticklabels()[0].set_fontweight("bold")
    ax.set_xlim(-0.6, len(sorted_methods) - 0.4)
    ax.set_ylim(108, -10)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=9)
    if ylabel:
        ax.set_ylabel("Rank Percentile (lower is better)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--out", default="693a8f224787a5a923946b4d/Figure/temporal_validation.pdf")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── TDL counts ────────────────────────────────────────────────────────────
    ldf = pd.read_csv(args.labels, sep="\t", dtype=str)
    ldf["old_n"] = ldf["idgTDL_old"].apply(normalize_tdl)
    ldf["new_n"] = ldf["idgTDL_new"].apply(normalize_tdl)
    tdl_order = ["Tclin", "Tchem", "Tbio", "Tdark"]
    old_c = {t: int((ldf["old_n"] == t).sum()) for t in tdl_order}
    new_c = {t: int((ldf["new_n"] == t).sum()) for t in tdl_order}

    data_A = load_task_data(args.results_dir, "upgrade_to_Tclin")
    data_B = load_task_data(args.results_dir, "upgrade_to_TclinOrTchem")

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8.2))
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[0.9, 0.85, 0.85],
                  width_ratios=[1.6, 1, 1],
                  hspace=0.55, wspace=0.30)

    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_pieO = fig.add_subplot(gs[0, 1])
    ax_pieN = fig.add_subplot(gs[0, 2])
    ax_D    = fig.add_subplot(gs[1, :])
    ax_E    = fig.add_subplot(gs[2, :])

    # ── A: bar chart ──────────────────────────────────────────────────────────
    x = np.arange(len(tdl_order))
    w = 0.32
    ax_bar.bar(x - w/2, [old_c[t] for t in tdl_order], width=w,
               color=[TDL_COLORS[t] for t in tdl_order],
               alpha=0.50, hatch="//", edgecolor="white", label="2021")
    ax_bar.bar(x + w/2, [new_c[t] for t in tdl_order], width=w,
               color=[TDL_COLORS[t] for t in tdl_order],
               alpha=0.95, label="2025")

    y_max = max(max(old_c.values()), max(new_c.values()))
    ax_bar.set_ylim(0, y_max * 1.55)  # extra headroom for badges + counts

    for i, t in enumerate(tdl_order):
        o, n = old_c[t], new_c[t]
        pct = (n - o) / o * 100
        sign = "+" if pct >= 0 else ""
        clr = "#2E7D32" if pct >= 0 else "#C62828"
        bar_top = max(o, n)

        # percentage badge — well above bars
        badge_y = bar_top + y_max * 0.18
        ax_bar.annotate(
            f"{sign}{pct:.1f}%", xy=(x[i], badge_y), ha="center", va="bottom",
            fontsize=9, color=clr, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=clr, lw=0.7))

        # count labels on top of bars — offset to avoid overlap
        ax_bar.text(x[i] - w/2, o + y_max * 0.02, f"{o:,}", ha="center",
                    va="bottom", fontsize=7, color="#666666")
        ax_bar.text(x[i] + w/2, n + y_max * 0.02, f"{n:,}", ha="center",
                    va="bottom", fontsize=7, color="#333333")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tdl_order, fontsize=11)
    ax_bar.set_ylabel("Number of Genes", fontsize=11)
    ax_bar.set_title("TDL Distribution: 2021 vs 2025", fontsize=11, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── B & C: donut charts ───────────────────────────────────────────────────
    def draw_pie(ax, counts, title, label_positions):
        total = sum(counts.values())
        sizes = [counts[t] for t in tdl_order]
        colors = [TDL_COLORS[t] for t in tdl_order]

        wedges, _, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct="", pctdistance=0.75, startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5))

        for j, (lx, ly, ha) in enumerate(label_positions):
            pct_val = sizes[j] / total * 100
            label = f"{tdl_order[j]}: {pct_val:.1f}%"
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=8.5,
                    color=TDL_COLORS[tdl_order[j]], fontweight="bold")

        ax.text(0, 0, f"Total\n{total:,}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333333")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=18)

    # Panel B (left pie): Tdark pulled inward to avoid collision with C
    pos_B = [
        (-0.30,  1.25, "center"),  # Tclin: top
        (-1.10,  0.20, "right"),   # Tchem: left
        ( 0.00, -1.20, "center"),  # Tbio: bottom
        ( 0.80,  0.85, "left"),    # Tdark: upper-right, pulled in
    ]
    # Panel C (right pie): Tchem pulled inward to avoid collision with B
    pos_C = [
        ( 0.30,  1.25, "center"),  # Tclin: top
        (-0.80,  0.85, "right"),   # Tchem: upper-left, pulled in
        ( 0.00, -1.20, "center"),  # Tbio: bottom
        ( 1.10,  0.20, "left"),    # Tdark: right
    ]
    draw_pie(ax_pieO, old_c, "2021 Distribution", pos_B)
    draw_pie(ax_pieN, new_c, "2025 Distribution", pos_C)

    # ── D & E: strip plots ────────────────────────────────────────────────────
    if data_A:
        n_up = len(list(data_A.values())[0])
        strip_plot(ax_D, data_A,
                   title=f"Upgraded to Tclin (from Tchem/Tbio/Tdark, n = {n_up})")
    if data_B:
        n_up = len(list(data_B.values())[0])
        strip_plot(ax_E, data_B,
                   title=f"Upgraded to Tclin/Tchem (from Tbio/Tdark, n = {n_up})")

    # ── panel labels ──────────────────────────────────────────────────────────
    for ax, label in [(ax_bar, "A"), (ax_pieO, "B"), (ax_pieN, "C"),
                      (ax_D, "D"), (ax_E, "E")]:
        ax.text(-0.06, 1.10, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="right")

    # ── save ──────────────────────────────────────────────────────────────────
    fig.subplots_adjust(right=0.96)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    svg_out = args.out.rsplit(".", 1)[0] + ".svg"
    plt.savefig(svg_out, format="svg", bbox_inches="tight")
    print(f"[DONE] saved -> {args.out}")
    print(f"[DONE] saved -> {svg_out}")


if __name__ == "__main__":
    main()
