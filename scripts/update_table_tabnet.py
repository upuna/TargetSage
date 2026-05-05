#!/usr/bin/env python3
"""Update Table 1 in main.tex with official pytorch-tabnet results."""
import os
import pandas as pd
import numpy as np

ROOT = "/home/zihend1/Genesis/TargetSage2"
CSV = f"{ROOT}/results/tabnet_official_all.csv"
TEX = f"{ROOT}/69d801ea1fd28f1005350c54/main.tex"

TASK_MAP = {
    "task_pharos_tclin_vs_others": "Clinical Targets",
    "task_pharos_tclin_tchem_vs_others": "Clinical & Chemical",
    "task_triage_tier1_vs_others": "Top-Tier Targets",
    "task_triage_tier12_vs_others": "High-Confidence Targets",
    "task_cancer_druggability": "Cancer-Relevant",
    "task_cancer_type_specific_target_prioritization": "Cancer Type-Specific",
    "task_pan_cancer_target_prioritization": "Pan-Cancer",
    "task_T1_targets_only": "T1 Cancer",
    "task_T1_T2_targets": "T1-T2 Cancer",
    "task_T1_T2_T3_targets": "T1-T3 Cancer",
    "task_sm_bucket1_vs_others": "Small-Molecule (Approved)",
    "task_sm_bucket123_vs_others": "Small-Molecule (Clinical+)",
    "task_ab_bucket1_vs_others": "Antibody (Approved)",
    "task_ab_bucket123_vs_others": "Antibody (Clinical+)",
    "task_protac_bucket1234_vs_others": "PROTAC Targets",
}

# Table row label → task key (for matching in tex)
ROW_LABELS = {
    "Clinical Targets": "task_pharos_tclin_vs_others",
    "Clinical \\& Chemical": "task_pharos_tclin_tchem_vs_others",
    "Top-Tier Targets": "task_triage_tier1_vs_others",
    "High-Confidence": "task_triage_tier12_vs_others",
    "Cancer-Relevant": "task_cancer_druggability",
    "Type-Specific": "task_cancer_type_specific_target_prioritization",
    "Pan-Cancer": "task_pan_cancer_target_prioritization",
    "T1 Cancer": "task_T1_targets_only",
    "T1--T2 Cancer": "task_T1_T2_targets",
    "T1--T3 Cancer": "task_T1_T2_T3_targets",
    "SM (Appr.)": "task_sm_bucket1_vs_others",
    "SM (Clin+)": "task_sm_bucket123_vs_others",
    "Ab (Appr.)": "task_ab_bucket1_vs_others",
    "Ab (Clin+)": "task_ab_bucket123_vs_others",
    "PROTAC": "task_protac_bucket1234_vs_others",
}

if not os.path.exists(CSV):
    print(f"ERROR: {CSV} not found. Run the official TabNet script first.")
    exit(1)

df = pd.read_csv(CSV)
print("Official TabNet results (÷100):")
print(f"{'Task':<30} {'Mean':>8} {'Std':>8}")
for _, row in df.iterrows():
    print(f"  {row['display']:<28} {row['mean']/100:>8.2f} {row['std']/100:>8.2f}")

# Compute macro-average
macro = df['mean'].mean() / 100
print(f"\nTabNet macro-average: {macro:.2f}%")

# Check if any TabNet value > Ours
# (hardcoded Ours values from table)
ours = {
    "task_pharos_tclin_vs_others": 11.2,
    "task_pharos_tclin_tchem_vs_others": 3.2,
    "task_triage_tier1_vs_others": 6.5,
    "task_triage_tier12_vs_others": 4.2,
    "task_cancer_druggability": 14.0,
    "task_cancer_type_specific_target_prioritization": 5.4,
    "task_pan_cancer_target_prioritization": 5.5,
    "task_T1_targets_only": 24.2,
    "task_T1_T2_targets": 7.8,
    "task_T1_T2_T3_targets": 5.4,
    "task_sm_bucket1_vs_others": 13.2,
    "task_sm_bucket123_vs_others": 9.1,
    "task_ab_bucket1_vs_others": 22.6,
    "task_ab_bucket123_vs_others": 17.5,
    "task_protac_bucket1234_vs_others": 17.2,
}

print("\nComparison (Official TabNet vs Ours):")
any_loss = False
for _, row in df.iterrows():
    tabnet_val = row['mean'] / 100
    ours_val = ours.get(row['task'], 0)
    win = "✓ Ours wins" if ours_val > tabnet_val else "✗ TabNet wins!"
    if ours_val <= tabnet_val:
        any_loss = True
    print(f"  {row['display']:<28} TabNet={tabnet_val:.2f} Ours={ours_val:.2f}  {win}")

if any_loss:
    print("\nWARNING: Some tasks where Ours doesn't win.")
else:
    print("\nOurs wins on all 15 tasks!")

print("\n[READY] To update main.tex, run: python3 update_table_tabnet.py --apply")

import sys
if "--apply" not in sys.argv:
    exit(0)

print("\nApplying updates to main.tex...")
with open(TEX, 'r') as f:
    tex = f.read()

# Build lookup: task → (mean_str, std_str)
tabnet_lookup = {}
for _, row in df.iterrows():
    m = row['mean'] / 100
    s = row['std'] / 100
    # Round to 1 decimal
    m1, s1 = round(m, 1), round(s, 1)
    tabnet_lookup[row['task']] = (m1, s1)

# The current table rows have pattern like:
# \some label & ... & $X.X_{\pm Y.Y}$ & ... & $\mathbf{Z.Z_{\pm W.W}}$ \\
# We need to replace the TabNet column (9th data column = position 9 in & split)

lines = tex.split('\n')
new_lines = []
for line in lines:
    # Find table rows by matching known row labels
    matched_task = None
    for label, task in ROW_LABELS.items():
        if line.strip().startswith(label + ' &'):
            matched_task = task
            break

    if matched_task and matched_task in tabnet_lookup:
        m1, s1 = tabnet_lookup[matched_task]
        # Parse the line: split by &
        parts = line.split('&')
        if len(parts) >= 11:
            # Column 9 (0-indexed 9) = TabNet
            old_tabnet = parts[9].strip()
            # Format new value
            new_val = f" ${m1:.1f}_{{\\pm {s1:.1f}}}$"
            parts[9] = new_val
            line = '&'.join(parts)
            print(f"  Updated {matched_task}: {old_tabnet.strip()} → {new_val.strip()}")

    new_lines.append(line)

# Update macro-average row
macro_rounded = round(macro, 1)
for i, line in enumerate(new_lines):
    if '\\textbf{Macro-Avg}' in line:
        parts = line.split('&')
        if len(parts) >= 11:
            old_tabnet = parts[9].strip()
            parts[9] = f" {macro_rounded}"
            new_lines[i] = '&'.join(parts)
            print(f"  Updated Macro-Avg TabNet: {old_tabnet} → {macro_rounded}")
        break

tex_new = '\n'.join(new_lines)
with open(TEX, 'w') as f:
    f.write(tex_new)

print(f"\n[DONE] Updated {TEX}")
print(f"New TabNet macro-avg: {macro_rounded}")
