#!/bin/bash
# Run agent features benchmark once extraction is complete
set -e
cd /home/zihend1/Genesis/TargetSage2
PYTHON=/home/zihend1/.conda/envs/py38/bin/python

echo "$(date): Starting agent benchmark pipeline"

# Step 1: Check extraction is complete
TOTAL=$(wc -l < results/agent_tools/agent_attributes.csv)
echo "$(date): agent_attributes.csv has $TOTAL lines (expected ~19033 incl. header)"

# Step 2: Filter attributes to common columns
echo "$(date): Filtering agent attributes..."
$PYTHON scripts/prepare_agent_attrs.py \
    --input  results/agent_tools/agent_attributes.csv \
    --output results/agent_tools/agent_attributes_filtered.csv \
    --min_coverage 0.05 \
    --max_cols 200 \
    2>&1 | tee results/agent_filter.log
echo "$(date): Filtering done"

# Step 3: Run TargetSage with agent-enriched attributes
echo "$(date): Training TargetSage with agent attributes..."
$PYTHON -u train.py \
    --scores results/agent_tools/agent_attributes_filtered.csv \
    --outdir results/train_agent \
    --attr_pca_dim 64 \
    2>&1 | tee results/train_agent.log
echo "$(date): Training done"

# Step 4: Print comparison
echo "$(date): Results comparison:"
echo "--- Original attributes (12-dim) ---"
tail -3 results/train_base/results.csv 2>/dev/null || echo "  (run train.py --outdir results/train_base first)"
echo "--- Agent attributes (filtered+PCA) ---"
tail -3 results/train_agent/results.csv 2>/dev/null || echo "  Not yet available"

echo "$(date): Agent benchmark complete!"
