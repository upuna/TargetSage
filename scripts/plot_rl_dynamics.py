#!/usr/bin/env python3
"""Plot GRPO training dynamics across 15 tasks: reward curves + policy stability."""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.linewidth": 0.8,
})

ROOT = "/home/zihend1/Genesis/TargetSage2"
V3 = f"{ROOT}/results/rl_grpo_v3/rl_log.csv"
MULTI = f"{ROOT}/results/grpo_multitask"

TASKS = {
    "Clinical Targets": V3,
    "Clinical & Chemical": f"{MULTI}/pharos_tclin_tchem/rl_log.csv",
    "Top-Tier":           f"{MULTI}/triage_tier1/rl_log.csv",
    "High-Confidence":    f"{MULTI}/triage_tier12/rl_log.csv",
    "Cancer-Relevant":    f"{MULTI}/cancer_druggability/rl_log.csv",
    "Type-Specific":      f"{MULTI}/cancer_type_specific/rl_log.csv",
    "Pan-Cancer":         f"{MULTI}/pan_cancer/rl_log.csv",
    "T1 Cancer":          f"{MULTI}/T1/rl_log.csv",
    "T1-T2 Cancer":       f"{MULTI}/T1_T2/rl_log.csv",
    "T1-T3 Cancer":       f"{MULTI}/T1_T2_T3/rl_log.csv",
    "SM (Appr.)":         f"{MULTI}/sm_bucket1/rl_log.csv",
    "SM (Clin+)":         f"{MULTI}/sm_bucket123/rl_log.csv",
    "Ab (Appr.)":         f"{MULTI}/ab_bucket1/rl_log.csv",
    "Ab (Clin+)":         f"{MULTI}/ab_bucket123/rl_log.csv",
    "PROTAC":             f"{MULTI}/protac/rl_log.csv",
}

CATEGORY = {
    "Clinical Targets": "PHAROS", "Clinical & Chemical": "PHAROS",
    "Top-Tier": "Triage", "High-Confidence": "Triage",
    "Cancer-Relevant": "Cancer", "Type-Specific": "Cancer", "Pan-Cancer": "Cancer",
    "T1 Cancer": "Cancer", "T1-T2 Cancer": "Cancer", "T1-T3 Cancer": "Cancer",
    "SM (Appr.)": "Modality", "SM (Clin+)": "Modality",
    "Ab (Appr.)": "Modality", "Ab (Clin+)": "Modality", "PROTAC": "Modality",
}
CAT_COLORS = {"PHAROS": "#5B4A8B", "Triage": "#3A7CA5",
              "Cancer": "#C47878", "Modality": "#2E8B6A"}


def load(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    data = {t: load(p) for t, p in TASKS.items()}
    data = {t: d for t, d in data.items() if d is not None and len(d) > 3}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # (A) Normalized reward curves (relative to bio baseline)
    axA = axes[0]
    for task, df in data.items():
        bio = df["bio_baseline"].iloc[0]
        rel = (df["global_reward"] / bio - 1) * 100
        c = CAT_COLORS[CATEGORY[task]]
        axA.plot(df["step"], rel, color=c, linewidth=1.1, alpha=0.75)
    axA.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    axA.set_xlabel("GRPO step")
    axA.set_ylabel(r"Reward uplift over bio baseline (%)")
    axA.set_title("(A) Reward improvement over training", fontweight="bold", fontsize=10.5)
    axA.grid(alpha=0.25)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # (B) Policy validity: valid_pct and reasoning_pct
    axB = axes[1]
    for task, df in data.items():
        c = CAT_COLORS[CATEGORY[task]]
        axB.plot(df["step"], df["valid_pct"], color=c, linewidth=1.0, alpha=0.55)
    axB.set_xlabel("GRPO step")
    axB.set_ylabel("Valid attribute output (%)")
    axB.set_ylim(70, 102)
    axB.set_title("(B) Policy stability", fontweight="bold", fontsize=10.5)
    axB.grid(alpha=0.25)
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # (C) PG loss: absolute, log scale if needed
    axC = axes[2]
    for task, df in data.items():
        c = CAT_COLORS[CATEGORY[task]]
        # Smooth with rolling mean for readability
        pg = df["pg_loss"].rolling(5, min_periods=1).mean()
        axC.plot(df["step"], pg, color=c, linewidth=1.0, alpha=0.65)
    axC.set_xlabel("GRPO step")
    axC.set_ylabel("PG loss (5-step moving avg)")
    axC.set_title("(C) Policy-gradient loss", fontweight="bold", fontsize=10.5)
    axC.grid(alpha=0.25)
    axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{ROOT}/69d801ea1fd28f1005350c54/figures/rl_dynamics.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".svg"), bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
