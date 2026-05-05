#!/usr/bin/env python3
"""Aggregate GRPO generalization results across all 15 tasks."""
import os, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/zihend1/Genesis/TargetSage2"
MULTI_DIR = f"{ROOT}/results/grpo_multitask"
V3_DIR = f"{ROOT}/results/rl_grpo_v3"  # Clinical Targets (already done)

# Task display names
TASK_DISPLAY = {
    "task_pharos_tclin_vs_others": "Clinical Targets",
    "task_pharos_tclin_tchem_vs_others": "Clinical & Chemical",
    "task_triage_tier1_vs_others": "Top-Tier",
    "task_triage_tier12_vs_others": "High-Confidence",
    "task_cancer_druggability": "Cancer-Relevant",
    "task_cancer_type_specific_target_prioritization": "Type-Specific",
    "task_pan_cancer_target_prioritization": "Pan-Cancer",
    "task_T1_targets_only": "T1 Cancer",
    "task_T1_T2_targets": "T1-T2 Cancer",
    "task_T1_T2_T3_targets": "T1-T3 Cancer",
    "task_sm_bucket1_vs_others": "SM (Appr.)",
    "task_sm_bucket123_vs_others": "SM (Clin+)",
    "task_ab_bucket1_vs_others": "Ab (Appr.)",
    "task_ab_bucket123_vs_others": "Ab (Clin+)",
    "task_protac_bucket1234_vs_others": "PROTAC",
}

SHORT = {
    "task_pharos_tclin_vs_others": None,  # v3_dir
    "task_pharos_tclin_tchem_vs_others": "pharos_tclin_tchem",
    "task_triage_tier1_vs_others": "triage_tier1",
    "task_triage_tier12_vs_others": "triage_tier12",
    "task_cancer_druggability": "cancer_druggability",
    "task_cancer_type_specific_target_prioritization": "cancer_type_specific",
    "task_pan_cancer_target_prioritization": "pan_cancer",
    "task_T1_targets_only": "T1",
    "task_T1_T2_targets": "T1_T2",
    "task_T1_T2_T3_targets": "T1_T2_T3",
    "task_sm_bucket1_vs_others": "sm_bucket1",
    "task_sm_bucket123_vs_others": "sm_bucket123",
    "task_ab_bucket1_vs_others": "ab_bucket1",
    "task_ab_bucket123_vs_others": "ab_bucket123",
    "task_protac_bucket1234_vs_others": "protac",
}


def load_result(task):
    short = SHORT.get(task)
    if short is None:
        # Clinical Targets: from v3 dir
        log_path = f"{V3_DIR}/rl_log.csv"
    else:
        log_path = f"{MULTI_DIR}/{short}/rl_log.csv"

    if not os.path.exists(log_path):
        return None

    df = pd.read_csv(log_path)
    if len(df) < 5:
        return None

    bio = df["bio_baseline"].iloc[0] * 100
    final_global = df["global_reward"].iloc[-1] * 100
    best_global = df["global_reward"].max() * 100
    return {
        "task": task,
        "display": TASK_DISPLAY[task],
        "bio_baseline": bio,
        "final_reward": final_global,
        "best_reward": best_global,
        "improvement_final": final_global - bio,
        "improvement_best": best_global - bio,
        "n_steps": len(df),
    }


def main():
    rows = []
    for task in TASK_DISPLAY:
        r = load_result(task)
        if r:
            rows.append(r)

    if not rows:
        print("No results yet.")
        return

    df = pd.DataFrame(rows)
    out_csv = f"{MULTI_DIR}/summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*90}")
    print(f"{'Task':<25} {'Bio-only':>12} {'+RL final':>12} {'+RL best':>12} {'Δ final':>10} {'Δ best':>10}")
    print(f"{'='*90}")
    for _, r in df.iterrows():
        print(f"{r['display']:<25} {r['bio_baseline']:>11.2f}% {r['final_reward']:>11.2f}% "
              f"{r['best_reward']:>11.2f}% {r['improvement_final']:>+9.2f}% {r['improvement_best']:>+9.2f}%")
    print(f"{'='*90}")
    print(f"{'MEAN':<25} {df['bio_baseline'].mean():>11.2f}% {df['final_reward'].mean():>11.2f}% "
          f"{df['best_reward'].mean():>11.2f}% {df['improvement_final'].mean():>+9.2f}% {df['improvement_best'].mean():>+9.2f}%")
    print(f"\nTasks completed: {len(df)}/15")
    print(f"Saved to {out_csv}")

    if len(df) >= 10:
        plot_rl_generalization(df)


CATEGORY = {
    "Clinical Targets": "PHAROS", "Clinical & Chemical": "PHAROS",
    "Top-Tier": "Triage", "High-Confidence": "Triage",
    "Cancer-Relevant": "Cancer", "Type-Specific": "Cancer", "Pan-Cancer": "Cancer",
    "T1 Cancer": "Cancer", "T1-T2 Cancer": "Cancer", "T1-T3 Cancer": "Cancer",
    "SM (Appr.)": "Modality", "SM (Clin+)": "Modality",
    "Ab (Appr.)": "Modality", "Ab (Clin+)": "Modality", "PROTAC": "Modality",
}
CAT_COLORS = {
    "PHAROS":   "#5B4A8B",
    "Triage":   "#3A7CA5",
    "Cancer":   "#C47878",
    "Modality": "#2E8B6A",
}


def plot_rl_generalization(df):
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })

    df = df.copy()
    df["category"] = df["display"].map(CATEGORY)
    df = df.sort_values(["category", "improvement_best"], ascending=[True, True]).reset_index(drop=True)

    tasks = df["display"].tolist()
    improvements = df["improvement_best"].values
    colors = [CAT_COLORS[df["category"].iloc[i]] for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(tasks))
    bars = ax.barh(y, improvements, color=colors, alpha=0.92,
                   edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels at bar ends
    x_max = max(improvements) * 1.15
    for i, v in enumerate(improvements):
        ax.text(v + x_max * 0.012, i, f"+{v:.1f}%", va="center", ha="left",
                fontsize=8.8, color="#333333", zorder=4)

    # Mean line
    mean_imp = df["improvement_best"].mean()
    ax.axvline(mean_imp, color="#444444", linestyle="--", linewidth=1.2,
               zorder=2, label=f"Mean  +{mean_imp:.2f}%")
    ax.axvline(0, color="black", linewidth=0.8, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(tasks, fontsize=10)
    ax.set_xlabel(r"$\Delta$ Adjusted F1 vs. bio-only (%)", fontsize=10.5)
    ax.set_xlim(0, x_max)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.25, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Category legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=k) for k, c in CAT_COLORS.items()]
    handles.append(plt.Line2D([0], [0], color="#444444", linestyle="--",
                              linewidth=1.2, label=f"Mean  +{mean_imp:.2f}%"))
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    plt.tight_layout()
    fig_path = f"{ROOT}/69d801ea1fd28f1005350c54/figures/rl_generalization.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.savefig(fig_path.replace(".pdf", ".svg"), bbox_inches="tight")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
