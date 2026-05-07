#!/usr/bin/env python3
"""2x2 ablation bar chart for GeneTrace main body."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.linewidth": 0.7,
})

BEST_COLOR = "#666666"
ALT_COLOR  = "#BBBBBB"

def bar_panel(ax, labels, values, highlight_idx, title, ylabel=False, ylim=(9.5, 11.5)):
    colors = [BEST_COLOR if i == highlight_idx else ALT_COLOR for i in range(len(labels))]
    ax.bar(range(len(labels)), values, color=colors,
           width=0.55, edgecolor="none", zorder=3)
    offset = (ylim[1] - ylim[0]) * 0.02
    for i, v in enumerate(values):
        ax.text(i, v + offset, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.5, color="#222222")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel("Macro-avg Adj. F1", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)

_mod_tasks = {
    "Bio":          [10.7,3.1,6.2,4.0,13.5,5.2,5.3,23.3,7.5,5.2,12.7,8.7,21.7,16.8,16.5],
    "Expl.":        [10.9,3.1,6.3,4.1,13.6,5.2,5.3,23.6,7.5,5.2,12.8,8.8,21.9,16.9,16.6],
    "Impl.":        [10.8,3.1,6.1,4.0,13.3,5.1,5.3,23.2,7.5,5.2,12.7,8.7,21.7,16.8,16.4],
    "Bio+\nExpl.":  [11.0,3.1,6.3,4.1,13.6,5.2,5.3,23.6,7.5,5.2,12.9,8.8,21.9,16.9,16.6],
    "Bio+\nImpl.":  [11.1,3.3,6.3,4.1,13.6,5.3,5.3,23.6,7.6,5.5,12.9,8.9,21.9,17.6,16.8],
    "Expl.+\nImpl.":[10.9,3.1,6.6,4.1,13.5,5.2,5.6,23.6,7.6,5.2,12.9,8.8,22.8,17.1,16.6],
    "Full":         [11.2,3.2,6.5,4.2,14.0,5.4,5.5,24.2,7.8,5.4,13.2,9.1,22.6,17.5,17.2],
}
_fus_tasks = {
    "Attn":  [9.8,3.0,5.7,3.6,11.8,4.5,4.8,21.1,6.8,4.7,11.4,7.8,19.8,15.2,16.1],
    "Concat":[11.3,3.3,6.1,3.9,14.1,4.9,5.1,22.6,7.2,5.0,13.3,8.5,22.7,17.6,15.4],
    "Sum":   [10.7,3.1,6.0,3.8,13.5,4.9,5.6,24.4,7.3,5.0,12.5,8.4,21.6,16.5,17.3],
    "Gated": [11.2,3.2,6.5,4.2,14.0,5.4,5.5,24.2,7.8,5.4,13.2,9.1,22.6,17.5,17.2],
}
_obj_tasks = {
    "BCE": [10.3,3.2,5.9,4.4,10.1,3.7,3.7,16.2,6.5,4.9,12.4,8.5,17.5,13.5,14.9],
    "uPU": [9.5,2.9,5.2,4.0,14.1,5.5,4.2,23.6,6.8,4.9,12.7,9.2,22.0,18.0,15.5],
    "nnPU":[11.2,3.2,6.5,4.2,14.0,5.4,5.5,24.2,7.8,5.4,13.2,9.1,22.6,17.5,17.2],
}
_rl_tasks = {
    "Static":[10.3,2.9,6.0,3.8,12.8,4.9,5.0,22.2,7.1,4.9,12.1,8.3,20.7,16.0,15.8],
    "Agent": [10.6,3.0,6.1,4.0,13.2,5.1,5.2,22.8,7.4,5.1,12.5,8.6,21.3,16.5,16.2],
    "+GRPO": [11.2,3.2,6.5,4.2,14.0,5.4,5.5,24.2,7.8,5.4,13.2,9.1,22.6,17.5,17.2],
}

def to_vals(d):
    return list(d.keys()), [np.mean(v) for v in d.values()]

mod_l, mod_v = to_vals(_mod_tasks)
fus_l, fus_v = to_vals(_fus_tasks)
obj_l, obj_v = to_vals(_obj_tasks)
rl_l,  rl_v  = to_vals(_rl_tasks)

fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.1))
fig.subplots_adjust(wspace=0.28, left=0.08, right=0.98, top=0.84, bottom=0.29)

bar_panel(axes[0], mod_l, mod_v, highlight_idx=6, title="Feature Modality",  ylabel=True, ylim=(10.4, 11.4))
bar_panel(axes[1], fus_l, fus_v, highlight_idx=3, title="Fusion Operator",              ylim=(9.3,  11.5))
bar_panel(axes[2], obj_l, obj_v, highlight_idx=2, title="Learning Objective",           ylim=(8.0,  11.8))
bar_panel(axes[3], rl_l,  rl_v,  highlight_idx=2, title="Attr. Extraction",             ylim=(9.7,  11.5))

for ax, lbl in zip(axes, ["A", "B", "C", "D"]):
    ax.text(-0.08, 1.0, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="right",
            color="#111111", clip_on=False)

out_dir = "69d801ea1fd28f1005350c54/figures"
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, "ablation_compact.pdf")
out_svg = os.path.join(out_dir, "ablation_compact.svg")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_svg, format="svg", bbox_inches="tight")
print(f"[DONE] {out_pdf}")
