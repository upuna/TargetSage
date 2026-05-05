#!/bin/bash
# Overnight automated pipeline
# 1. Wait for agent collect to finish
# 2. Run fact check
# 3. Run LLM attribute extraction on agent profiles
# 4. Compare agent vs original features

set -e
cd /home/zihend1/Genesis/TargetSage2
PYTHON=/home/zihend1/.conda/envs/py38/bin/python

echo "$(date): Starting overnight pipeline"

# Step 1: Wait for agent collect to finish
echo "$(date): Waiting for agent collect..."
while ps aux | grep "agent_tools.py" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done
echo "$(date): Agent collect finished"

# Step 2: Fact check
echo "$(date): Running fact check..."
$PYTHON -u scripts/agent_tools.py --stage check --outdir results/agent_tools \
    > results/overnight_factcheck.log 2>&1
echo "$(date): Fact check done"
cat results/overnight_factcheck.log

# Step 3: Extract attributes with GPT-4o-mini
echo "$(date): Starting LLM attribute extraction..."
$PYTHON -u scripts/agent_tools.py --stage extract --outdir results/agent_tools \
    > results/overnight_extract.log 2>&1 &
EXTRACT_PID=$!
echo "$(date): Extract started (PID: $EXTRACT_PID)"

# Wait for extraction
wait $EXTRACT_PID
echo "$(date): Extraction done"

# Step 4: Summary stats
echo "$(date): Computing summary statistics..."
$PYTHON -c "
import pandas as pd
import json

# Agent collection stats
records = []
with open('results/agent_tools/agent_collected.jsonl') as f:
    for line in f:
        records.append(json.loads(line))

print(f'Total genes collected: {len(records)}')
print(f'Avg sources per gene: {sum(r[\"n_sources\"] for r in records)/len(records):.1f}')
print(f'Genes with leakage: {sum(1 for r in records if r.get(\"leakage_detected\"))}')
print(f'Genes with conflicts: {sum(1 for r in records if r.get(\"conflicts\"))}')

source_counts = {}
for r in records:
    for src, has in r['sources'].items():
        if has:
            source_counts[src] = source_counts.get(src, 0) + 1
print(f'\nSource coverage:')
for src, count in sorted(source_counts.items(), key=lambda x:-x[1]):
    print(f'  {src}: {count}/{len(records)} ({count/len(records)*100:.0f}%)')

# Agent attribute stats (if available)
try:
    attrs = pd.read_csv('results/agent_tools/agent_attributes.csv')
    attr_cols = [c for c in attrs.columns if c != 'Gene_Symbol']
    print(f'\nAgent attributes: {len(attrs)} genes, {len(attr_cols)} unique dimensions')
    print(f'Top attributes: {attr_cols[:15]}')
except:
    print('\nAgent attributes not yet available')
" > results/overnight_stats.log 2>&1
cat results/overnight_stats.log

echo "$(date): Overnight pipeline complete!"
