#!/usr/bin/env python3
import glob, os, pandas as pd, numpy as np, json
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score

def get_unlabeled(csv, positives, ct_genes):
    df = pd.read_csv(csv)
    df = df.sort_values("rank").reset_index(drop=True) if "rank" in df.columns else df.reset_index(drop=True)
    score_col = [c for c in df.columns if "score" in c.lower()][0]
    df = df.rename(columns={score_col: "score"})
    df["is_positive"] = df["Gene_Symbol"].isin(positives)
    df["is_ct"] = df["Gene_Symbol"].str.upper().isin(ct_genes)
    return df[~df["is_positive"]].copy().reset_index(drop=True)

def or_at(ul, pct):
    M, K = len(ul), ul["is_ct"].sum()
    n = max(1, int(M * pct / 100))
    ct_top = ul.iloc[:n]["is_ct"].sum()
    ct_rest = K - ct_top
    table = [[ct_top, n - ct_top], [ct_rest, M - n - ct_rest]]
    return fisher_exact(table, alternative="greater")[0]

label_df = pd.read_csv("data/gene_labels.tsv", sep="\t")
label_df["task_pharos_tclin_vs_others"] = pd.to_numeric(
    label_df["task_pharos_tclin_vs_others"], errors="coerce").fillna(0).astype(int)
positives = set(label_df[label_df["task_pharos_tclin_vs_others"] == 1]["Gene_Symbol"].astype(str))
ct_df = pd.read_csv("data/chembl_clinical_targets_phase23.csv")
ct_genes = set(ct_df["gene_symbol"].astype(str).str.upper())

rows = []

# baseline
csv = "results/inference_20260429_102203/pharos_tclin_vs_others_ranking.csv"
ul = get_unlabeled(csv, positives, ct_genes)
rows.append({"config": "baseline(pi=0.10,ep=30,b=0.6)",
             "OR@1%": or_at(ul, 1), "OR@5%": or_at(ul, 5), "AUROC": roc_auc_score(ul["is_ct"].astype(int), ul["score"])})

# tune runs
for d in sorted(glob.glob("results/tune/inference_*")):
    cfg_path = os.path.join(d, "config.json")
    csv = os.path.join(d, "pharos_tclin_vs_others_ranking.csv")
    if not os.path.exists(csv):
        continue
    cfg = json.load(open(cfg_path))
    ul = get_unlabeled(csv, positives, ct_genes)
    label = f"pi={cfg['pi_cap']},ep={cfg['nnpu_epochs']},b={cfg['beta']}"
    rows.append({"config": label,
                 "OR@1%": or_at(ul, 1), "OR@5%": or_at(ul, 5),
                 "AUROC": roc_auc_score(ul["is_ct"].astype(int), ul["score"])})

df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.float_format", "{:.3f}".format)
print(df.to_string(index=False))
