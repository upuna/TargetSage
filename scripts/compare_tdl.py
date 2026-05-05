#!/usr/bin/env python3
import glob, os, pandas as pd, numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

def get_ul_tdl(csv, positives, ct_genes, tdl_set, label_df):
    df = pd.read_csv(csv)
    df = df.sort_values("rank").reset_index(drop=True) if "rank" in df.columns else df.reset_index(drop=True)
    sc = [c for c in df.columns if "score" in c.lower()][0]
    df = df.rename(columns={sc: "score"})
    df["is_positive"] = df["Gene_Symbol"].isin(positives)
    df["is_ct"] = df["Gene_Symbol"].str.upper().isin(ct_genes)
    tdl_map = dict(zip(label_df["Gene_Symbol"], label_df["idgTDL_new"].str.upper()))
    df["tdl"] = df["Gene_Symbol"].map(tdl_map)
    return df[~df["is_positive"] & df["tdl"].isin(tdl_set)].copy().reset_index(drop=True)

def or_at(ul, p):
    M, K = len(ul), ul["is_ct"].sum()
    n = max(1, int(M * p / 100))
    ct = ul.iloc[:n]["is_ct"].sum()
    return fisher_exact([[ct, n - ct], [K - ct, M - n - (K - ct)]], alternative="greater")[0]

label_df = pd.read_csv("data/gene_labels.tsv", sep="\t")
label_df["task_pharos_tclin_vs_others"] = pd.to_numeric(
    label_df["task_pharos_tclin_vs_others"], errors="coerce").fillna(0).astype(int)
pos = set(label_df[label_df["task_pharos_tclin_vs_others"] == 1]["Gene_Symbol"].astype(str))
ct = set(pd.read_csv("data/chembl_clinical_targets_phase23.csv")["gene_symbol"].astype(str).str.upper())

ts_csv = "results/tune/inference_20260429_111532/pharos_tclin_vs_others_ranking.csv"

for tdl_filter, label in [
    ({"TBIO"}, "TBIO only"),
    ({"TBIO", "TDARK"}, "TBIO+TDARK"),
    ({"TBIO", "TDARK", "TCHEM"}, "All unlabeled"),
]:
    print("\n=== " + label + " ===")
    ul = get_ul_tdl(ts_csv, pos, ct, tdl_filter, label_df)
    print("n=%d, ct_hits=%d" % (len(ul), ul["is_ct"].sum()))
    r1, r5, au = or_at(ul, 1), or_at(ul, 5), roc_auc_score(ul["is_ct"].astype(int), ul["score"])
    print("%-16s  %.2f    %.2f    %.4f  <<" % ("TargetSage(best)", r1, r5, au))
    print("-" * 55)
    for csv in sorted(glob.glob("results/baseline_inference_2025/*_ranking.csv")):
        name = csv.split("/")[-1].replace("_ranking.csv", "")
        ul2 = get_ul_tdl(csv, pos, ct, tdl_filter, label_df)
        r1b, r5b, aub = or_at(ul2, 1), or_at(ul2, 5), roc_auc_score(ul2["is_ct"].astype(int), ul2["score"])
        print("%-16s  %.2f    %.2f    %.4f" % (name, r1b, r5b, aub))
