#!/bin/bash
# Overnight parallel experiment: run 8 configs across 6 GPUs
# Each config runs 50 seeds sequentially on its assigned GPU
# 8 configs / 6 GPUs = ~2 waves, total time ~1.5-2 hours

set -e
cd /home/zihend1/Genesis/TargetSage2
PYTHON=/home/zihend1/.conda/envs/py38/bin/python
LABELS=data/gene_labels_2021.tsv
TASKS="task_pharos_tclin_vs_others,task_pharos_tclin_tchem_vs_others"
OUTBASE=results/overnight_sweep
mkdir -p $OUTBASE

# Function to run one config on one GPU
run_config() {
    local NAME=$1
    local GPU=$2
    local D_LATENT=$3
    local HEAD_H=$4
    local EMB_PCA=$5
    local WARMUP=$6
    local NNPU=$7
    local LR=$8
    local DROPOUT=$9
    local BETA=${10}
    local N_SEEDS=${11}

    local CFGDIR=$OUTBASE/$NAME
    mkdir -p $CFGDIR

    echo "[$(date +%H:%M:%S)] Starting config=$NAME on GPU=$GPU (${N_SEEDS} seeds)"

    for SEED in $(seq 0 $((N_SEEDS-1))); do
        # Skip if this seed's output already exists
        DONE_COUNT=$(find $CFGDIR -name "pharos_tclin_vs_others_ranking.csv" 2>/dev/null | wc -l)
        if [ $DONE_COUNT -ge $N_SEEDS ]; then
            echo "  Config $NAME already complete ($DONE_COUNT runs)"
            break
        fi

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON inference.py \
            --labels $LABELS \
            --task $TASKS \
            --seed $SEED \
            --d_latent $D_LATENT \
            --head_h $HEAD_H \
            --emb_pca_dim $EMB_PCA \
            --warmup_epochs $WARMUP \
            --nnpu_epochs $NNPU \
            --lr $LR \
            --dropout $DROPOUT \
            --beta $BETA \
            --outdir $CFGDIR \
            > /dev/null 2>&1

        if [ $(( (SEED+1) % 10 )) -eq 0 ]; then
            echo "  [$NAME] seed $SEED done"
        fi
    done
    echo "[$(date +%H:%M:%S)] Finished config=$NAME"
}

# Wave 1: 6 configs on 6 GPUs
run_config "default"      0 256 512  256 10 30 2e-4 0.2 0.6 50 &
run_config "large"        1 512 1024 256 10 30 2e-4 0.2 0.6 50 &
run_config "long"         2 256 512  256 25 50 2e-4 0.2 0.6 50 &
run_config "fullEmb"      3 256 512  0   10 30 2e-4 0.2 0.6 50 &
run_config "large_long"   4 512 1024 256 25 50 1e-4 0.15 0.6 50 &
run_config "fullEmb_large" 5 512 1024 0  15 40 2e-4 0.2 0.6 50 &

wait
echo "[$(date +%H:%M:%S)] Wave 1 done"

# Wave 2: 2 more configs
run_config "lowDrop_hiBeta" 0 256 512 256 15 40 2e-4 0.1 0.8 50 &
run_config "long_fullEmb"   1 256 512 0   25 50 2e-4 0.2 0.6 50 &

wait
echo "[$(date +%H:%M:%S)] Wave 2 done"

# Phase 2: Ensemble and evaluate
echo ""
echo "===== ENSEMBLING AND EVALUATION ====="
$PYTHON scripts/overnight_ensemble.py

echo "[$(date +%H:%M:%S)] ALL DONE"
