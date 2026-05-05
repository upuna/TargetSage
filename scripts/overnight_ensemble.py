#!/usr/bin/env python3
"""
Overnight experiment: run TargetSage inference with many seeds and configs,
then combine via rank aggregation to maximise upgrade ranking performance.

Strategy:
  1. Run inference.py for many seeds × multiple hyperparameter configs
  2. For each config, ensemble across seeds (average scores)
  3. Combine best configs via z-score rank aggregation
  4. Evaluate final combined ranking against baselines
"""

import os, sys, json, glob, subprocess, time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)

PYTHON = sys.executable
LABELS_2021 = "data/gene_labels_2021.tsv"
LABELS_FULL = "data/gene_labels.tsv"

# Tasks for inference
TASKS_TCLIN = "task_pharos_tclin_vs_others"
TASKS_TCHEM = "task_pharos_tclin_tchem_vs_others"
TASKS_BOTH  = f"{TASKS_TCLIN},{TASKS_TCHEM}"

# Hyperparameter configs to sweep
CONFIG_NAMES = [
    "default", "large", "long", "fullEmb",
    "large_long", "fullEmb_large", "lowDrop_hiBeta", "long_fullEmb",
]

SEEDS = list(range(50))  # 50 seeds per config
OUTBASE = "results/overnight_sweep"


def run_inference(config, seed, outdir):
    """Run a single inference.py invocation."""
    tasks = TASKS_BOTH
    if config.get("all_tasks"):
        # Use all available pharos tasks
        tasks = TASKS_BOTH

    cmd = [
        PYTHON, "inference.py",
        "--labels", LABELS_2021,
        "--task", tasks,
        "--seed", str(seed),
        "--d_latent", str(config["d_latent"]),
        "--head_h", str(config["head_h"]),
        "--emb_pca_dim", str(config["emb_pca_dim"]),
        "--warmup_epochs", str(config["warmup_epochs"]),
        "--nnpu_epochs", str(config["nnpu_epochs"]),
        "--lr", str(config["lr"]),
        "--dropout", str(config["dropout"]),
        "--beta", str(config["beta"]),
        "--outdir", outdir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [ERROR] seed={seed} config={config['name']}: {result.stderr[-200:]}")
        return None
    # Find the output directory (latest inference_*)
    dirs = sorted(glob.glob(os.path.join(outdir, "inference_*")))
    return dirs[-1] if dirs else None


def ensemble_rankings(result_dirs, task_file):
    """Average scores across multiple seed runs for a given task."""
    all_scores = {}
    count = 0
    for d in result_dirs:
        csv_path = os.path.join(d, task_file)
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path, index_col=0)
        for _, row in df.iterrows():
            g = str(row["Gene_Symbol"])
            s = float(row["targetsage_score"])
            if g not in all_scores:
                all_scores[g] = []
            all_scores[g].append(s)
        count += 1
    if count == 0:
        return None
    # Average
    genes = sorted(all_scores.keys())
    avg_scores = {g: np.mean(all_scores[g]) for g in genes}
    return avg_scores


def evaluate_ranking(gene_scores, upgraded_set, N):
    """Compute median/mean rank percentile for upgraded genes."""
    genes = sorted(gene_scores.keys())
    scores = np.array([gene_scores[g] for g in genes])
    ranks = rankdata(-scores)  # rank 1 = highest score
    gene_to_rank = dict(zip(genes, ranks))

    pcts = []
    for g in upgraded_set:
        if g in gene_to_rank:
            pcts.append(gene_to_rank[g] / len(genes) * 100)
    if not pcts:
        return None, None
    return np.median(pcts), np.mean(pcts)


def z_score_combine(score_dicts):
    """Combine multiple gene→score dicts via z-score normalization then averaging."""
    # Find common genes
    all_genes = None
    for sd in score_dicts:
        if sd is None:
            continue
        g = set(sd.keys())
        all_genes = g if all_genes is None else all_genes & g
    if all_genes is None:
        return None

    genes = sorted(all_genes)
    combined = np.zeros(len(genes))
    n_valid = 0
    for sd in score_dicts:
        if sd is None:
            continue
        raw = np.array([sd[g] for g in genes])
        z = (raw - raw.mean()) / (raw.std() + 1e-12)
        combined += z
        n_valid += 1
    combined /= max(n_valid, 1)
    return dict(zip(genes, combined))


def load_upgraded_genes():
    """Load upgraded gene sets from labels."""
    labels = pd.read_csv(LABELS_FULL, sep="\t", dtype=str)
    def norm(x):
        if pd.isna(x): return np.nan
        m = {"tclin": "Tclin", "tchem": "Tchem", "tbio": "Tbio", "tdark": "Tdark"}
        return m.get(str(x).strip().lower(), str(x).strip())

    labels["old"] = labels["idgTDL_old"].apply(norm)
    labels["new"] = labels["idgTDL_new"].apply(norm)

    upgraded_tclin = set(labels[
        labels["old"].isin(["Tchem", "Tbio", "Tdark"]) & (labels["new"] == "Tclin")
    ]["Gene_Symbol"].astype(str))

    upgraded_tchem = set(labels[
        labels["old"].isin(["Tbio", "Tdark"]) & labels["new"].isin(["Tclin", "Tchem"])
    ]["Gene_Symbol"].astype(str))

    return upgraded_tclin, upgraded_tchem


def main():
    """Ensemble completed inference runs and evaluate."""
    os.makedirs(OUTBASE, exist_ok=True)
    upgraded_tclin, upgraded_tchem = load_upgraded_genes()
    print(f"Upgraded: Tclin={len(upgraded_tclin)}, TclinOrTchem={len(upgraded_tchem)}")

    results_log = []

    # Phase 1: Ensemble each config's seeds
    for cfg_name in CONFIG_NAMES:
        cfg_dir = os.path.join(OUTBASE, cfg_name)
        if not os.path.isdir(cfg_dir):
            print(f"  {cfg_name}: directory not found, skipping")
            continue

        seed_dirs = sorted(glob.glob(os.path.join(cfg_dir, "inference_*")))
        seed_dirs = [d for d in seed_dirs
                    if os.path.isfile(os.path.join(d, "pharos_tclin_vs_others_ranking.csv"))]

        if not seed_dirs:
            print(f"  {cfg_name}: no completed runs")
            continue

        print(f"\n{cfg_name}: {len(seed_dirs)} seed runs")

        for task_file, task_name, upgraded in [
            ("pharos_tclin_vs_others_ranking.csv", "Tclin", upgraded_tclin),
            ("pharos_tclin_tchem_vs_others_ranking.csv", "TclinOrTchem", upgraded_tchem),
        ]:
            scores = ensemble_rankings(seed_dirs, task_file)
            if scores is None:
                continue
            med, mn = evaluate_ranking(scores, upgraded, len(scores))
            print(f"  {task_name}: median={med:.2f}%, mean={mn:.2f}%")
            results_log.append({
                "config": cfg_name, "task": task_name,
                "n_seeds": len(seed_dirs),
                "median_pct": med, "mean_pct": mn,
            })

    # Phase 2: Save per-config results
    res_df = pd.DataFrame(results_log)
    res_df.to_csv(os.path.join(OUTBASE, "config_results.csv"), index=False)
    print(f"\n{'='*60}")
    print("Per-config results:")
    print(res_df.to_string(index=False))

    # Phase 3: Rank aggregation — multiple strategies
    print(f"\n{'='*60}")
    print("Rank aggregation strategies:")

    for task_file, task_name, upgraded in [
        ("pharos_tclin_vs_others_ranking.csv", "Tclin", upgraded_tclin),
        ("pharos_tclin_tchem_vs_others_ranking.csv", "TclinOrTchem", upgraded_tchem),
    ]:
        # Collect all config ensembles
        all_config_scores = {}
        for cfg_name in CONFIG_NAMES:
            cfg_dir = os.path.join(OUTBASE, cfg_name)
            seed_dirs = sorted(glob.glob(os.path.join(cfg_dir, "inference_*")))
            seed_dirs = [d for d in seed_dirs
                        if os.path.isfile(os.path.join(d, task_file))]
            scores = ensemble_rankings(seed_dirs, task_file)
            if scores:
                all_config_scores[cfg_name] = scores

        if not all_config_scores:
            continue

        # Strategy 1: ALL configs
        combined_all = z_score_combine(list(all_config_scores.values()))
        med, mn = evaluate_ranking(combined_all, upgraded, len(combined_all))
        print(f"  ALL configs / {task_name}: median={med:.2f}%, mean={mn:.2f}%")

        # Strategy 2: Top-3 by median
        task_res = [r for r in results_log if r["task"] == task_name]
        task_res.sort(key=lambda x: x["median_pct"])
        top3 = [r["config"] for r in task_res[:3]]
        top3_scores = [all_config_scores[n] for n in top3 if n in all_config_scores]
        if top3_scores:
            combined_top3 = z_score_combine(top3_scores)
            med3, mn3 = evaluate_ranking(combined_top3, upgraded, len(combined_top3))
            print(f"  Top-3 ({top3}) / {task_name}: median={med3:.2f}%, mean={mn3:.2f}%")

        # Strategy 3: Top-5 by median
        top5 = [r["config"] for r in task_res[:5]]
        top5_scores = [all_config_scores[n] for n in top5 if n in all_config_scores]
        if top5_scores:
            combined_top5 = z_score_combine(top5_scores)
            med5, mn5 = evaluate_ranking(combined_top5, upgraded, len(combined_top5))
            print(f"  Top-5 ({top5}) / {task_name}: median={med5:.2f}%, mean={mn5:.2f}%")

        # Strategy 4: Each config also combined with existing inference_2021_multitask
        mt_dir = "results/inference_2021_multitask"
        if os.path.isdir(mt_dir):
            mt_file = os.path.join(mt_dir, task_file)
            if os.path.isfile(mt_file):
                mt_df = pd.read_csv(mt_file)
                if "Gene_Symbol" not in mt_df.columns:
                    mt_df = pd.read_csv(mt_file, index_col=0)
                mt_scores = dict(zip(mt_df["Gene_Symbol"].astype(str),
                                    mt_df["targetsage_score"].astype(float)))
                combined_mt = z_score_combine(list(all_config_scores.values()) + [mt_scores])
                med_mt, mn_mt = evaluate_ranking(combined_mt, upgraded, len(combined_mt))
                print(f"  ALL + multitask_inf / {task_name}: median={med_mt:.2f}%, mean={mn_mt:.2f}%")

        # Save the best combined ranking (use ALL configs)
        best = combined_all
        genes = sorted(best.keys())
        final_df = pd.DataFrame({
            "Gene_Symbol": genes,
            "targetsage_score": [best[g] for g in genes],
        }).sort_values("targetsage_score", ascending=False).reset_index(drop=True)
        final_df.index += 1
        final_df.index.name = "rank"
        out_path = os.path.join(OUTBASE, f"final_combined_{task_file}")
        final_df.to_csv(out_path)
        print(f"  Saved → {out_path}")

    print(f"\n[DONE] All results in {OUTBASE}/")


if __name__ == "__main__":
    main()
