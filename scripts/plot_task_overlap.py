#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute and plot Jaccard similarity matrix across 15 benchmark tasks."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

LABELS_FILE = "data/gene_labels.tsv"
OUT_DIR     = "69d801ea1fd28f1005350c54/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Column name in gene_labels.tsv -> display name (paper order)
TASKS = [
    ("task_pharos_tclin_vs_others",                     "Clinical\nTargets"),
    ("task_pharos_tclin_tchem_vs_others",               "Clinical &\nChemical"),
    ("task_triage_tier1_vs_others",                     "Top-Tier\nTargets"),
    ("task_triage_tier12_vs_others",                    "High-\nConfidence"),
    ("task_cancer_druggability",                        "Cancer-\nRelevant"),
    ("task_cancer_type_specific_target_prioritization", "Type-\nSpecific"),
    ("task_pan_cancer_target_prioritization",           "Pan-\nCancer"),
    ("task_T1_targets_only",                            "T1\nCancer"),
    ("task_T1_T2_targets",                              "T1-T2\nCancer"),
    ("task_T1_T2_T3_targets",                           "T1-T3\nCancer"),
    ("task_sm_bucket1_vs_others",                       "SM\n(Appr.)"),
    ("task_sm_bucket123_vs_others",                     "SM\n(Clin+)"),
    ("task_ab_bucket1_vs_others",                       "Ab\n(Appr.)"),
    ("task_ab_bucket123_vs_others",                     "Ab\n(Clin+)"),
    ("task_protac_bucket1234_vs_others",                "PROTAC"),
]

CATEGORY_SPANS = [
    (0,  2,  "#1565C0", "PHAROS (Disease-agnostic)"),
    (2,  4,  "#E65100", "Triage (Disease-agnostic)"),
    (4,  10, "#2E7D32", "Cancer Druggability (Domain-specific)"),
    (10, 15, "#C62828", "Drug Modality (Domain-specific)"),
]

# ── Load full gene label matrix ──────────────────────────────────────────────
df = pd.read_csv(LABELS_FILE, sep="\t")
cols   = [c for c, _ in TASKS]
labels = [lab for _, lab in TASKS]
positives = [set(df.loc[df[c] == 1, "Gene_Symbol"]) for c in cols]

for lab, p in zip(labels, positives):
    print(f"{lab.replace(chr(10), ' '):25s}  |P|={len(p)}")

n = len(positives)

# ── Jaccard matrix ───────────────────────────────────────────────────────────
J = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        inter = len(positives[i] & positives[j])
        union = len(positives[i] | positives[j])
        J[i, j] = inter / union if union > 0 else 0.0

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(9.5, 8.5))
fig.subplots_adjust(left=0.18, right=0.84, top=0.92, bottom=0.20)

cmap = LinearSegmentedColormap.from_list("wpu", ["#FFFFFF", "#6E5C7A"])
im = ax.imshow(J, cmap=cmap, vmin=0, vmax=1, aspect="auto")

# Cell annotations
for i in range(n):
    for j in range(n):
        v = J[i, j]
        color = "white" if v > 0.55 else "#333333"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=7, color=color)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, fontsize=8, rotation=40, ha="right",
                   rotation_mode="anchor")
ax.set_yticklabels(labels, fontsize=8)

# Color tick labels by category
tick_colors = {}
for start, end, color, _ in CATEGORY_SPANS:
    for i in range(start, end):
        tick_colors[i] = color
for tick, idx in zip(ax.get_xticklabels(), range(n)):
    tick.set_color(tick_colors[idx])
for tick, idx in zip(ax.get_yticklabels(), range(n)):
    tick.set_color(tick_colors[idx])

# Category divider lines — thick and solid
for start, end, color, _ in CATEGORY_SPANS:
    for xy in [start - 0.5, end - 0.5]:
        ax.axhline(xy, color=color, linewidth=2.0, alpha=1.0)
        ax.axvline(xy, color=color, linewidth=2.0, alpha=1.0)

# Colored margin bars along top and left axes to mark categories
bar_w = 0.35
for start, end, color, _ in CATEGORY_SPANS:
    mid = (start + end) / 2 - 0.5
    span = end - start
    # top bar
    ax.add_patch(mpatches.FancyArrowPatch(
        (start - 0.5, -1.7), (end - 0.5, -1.7),
        arrowstyle='-', color=color, linewidth=5, clip_on=False))
    # left bar
    ax.add_patch(mpatches.FancyArrowPatch(
        (-2.4, start - 0.5), (-2.4, end - 0.5),
        arrowstyle='-', color=color, linewidth=5, clip_on=False))

# Legend
patches = [mpatches.Patch(facecolor=c, edgecolor="none",
                          label=lab)
           for _, _, c, lab in CATEGORY_SPANS]
ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1.0),
          fontsize=8, frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
          borderaxespad=0, handlelength=1.5, handleheight=1.4)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("Jaccard similarity", fontsize=8)
cbar.ax.tick_params(labelsize=7)

ax.set_title("Positive-set overlap across 15 benchmark tasks",
             fontsize=10, pad=12)

out_pdf = os.path.join(OUT_DIR, "task_overlap.pdf")
out_svg = os.path.join(OUT_DIR, "task_overlap.svg")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, format="svg", bbox_inches="tight")
print(f"\n[DONE] {out_pdf}")

# ── Cross-category overlap stats ─────────────────────────────────────────────
print("\nCross-category Jaccard (max / mean):")
cats = [(s, e, name.replace("\n", " ")) for s, e, _, name in CATEGORY_SPANS]
for i, (s1, e1, n1) in enumerate(cats):
    for j, (s2, e2, n2) in enumerate(cats):
        if j <= i:
            continue
        block = J[s1:e1, s2:e2]
        print(f"  {n1:35s} x {n2:35s}  max={block.max():.3f}  mean={block.mean():.3f}")
