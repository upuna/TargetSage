# TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning

Three-module framework for prioritizing human protein-coding genes as drug targets across 19,032 genes and 15 benchmark tasks.

```
M1 — Agentic Profiling
  6-tool pipeline (NCBI · UniProt · PubMed · Open Targets · ...)
  → features_llm_structured_scores.csv   (23-dim LLM attribute scores)
  → features_llm_embedding.csv           (1536-dim text embeddings)
        ↓
M2 — Guided Reasoning (GRPO)
  Fine-tunes reasoning policy via RL reward = Adjusted F1 of frozen M3 proxy
        ↓
M3 — PU-Aware Scoring
  ┌─── Bio features          (482-dim)  ─┐
  ├─── LLM attribute scores  ( 23-dim)  ─┤→ Gated fusion → nnPU discriminator
  └─── LLM embeddings        (256-dim)  ─┘
        ↓
  train.py      — 15-task benchmark evaluation
  inference.py  — full-genome ranking
```

M1 and M2 outputs are precomputed in `data/`. Reviewers only need to run M3.

---

## Quick Start

```bash
pip install -r requirements.txt

# Quick test: 2 tasks, 3 seeds (~5 min on CPU)
python train.py --tasks task_pharos_tclin_vs_others,task_T1_targets_only --seeds 0,1,2

# Full 15-task reproduction (~3 hours on RTX 3090)
python train.py
```

Results saved to `results/train_<timestamp>/results.csv`.

---

## Baselines

```bash
# All 10 baselines (7 classical + ResNet / TabNet / FT-Transformer)
python scripts/run_baselines.py

# Classical only (fast, CPU)
python scripts/run_baselines.py --mode classical

# Deep tabular only
python scripts/run_baselines.py --mode deep
```

---

## Inference

Rank all ~19k genes for a given task:

```bash
python inference.py --task task_pharos_tclin_vs_others

# Save top-500 rankings
python inference.py --task task_pharos_tclin_vs_others --top_k 500
```

---

## Repository Structure

```
TargetSage/
├── train.py                         M3: train and evaluate across 15 tasks
├── inference.py                     M3: full-genome ranking
├── targetsage/
│   ├── model.py                     3-head gated-fusion MLP
│   ├── loss.py                      nnPU loss (Kiryo 2017) + semantic reweighting
│   ├── metrics.py                   Adjusted F1 = R_soft² / p̄
│   └── data.py                      Data loaders and feature matrix builder
├── scripts/
│   ├── m1_agentic_profiling.py      M1 documented stub + prompt template
│   ├── m2_grpo_training.py          M2 GRPO fine-tuning pipeline
│   └── run_baselines.py             All 10 baselines
└── data/
    ├── features_llm_structured_scores.csv   19014 genes × 23 LLM attribute scores
    ├── gene_labels.tsv                       19032 genes × 15 binary task labels
    ├── gene_labels_2021.tsv                  2021 snapshot for temporal validation
    ├── gene_summaries.tsv                    NCBI Gene summary text
    └── chembl_clinical_targets_phase23.csv   ChEMBL Phase 2/3 clinical candidates
```

> `data/gene_features.tsv` (482-dim structured bio features) and
> `data/features_llm_embedding.csv` (1536-dim embeddings) are large files not
> tracked by git. See `data/README.md` for sources and preparation instructions.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.6 | Hybrid prior: `π = α·π_data + (1-α)·π_llm` |
| `--beta` | 0.6 | Semantic reweighting: `w = β·D(z) + (1-β)·cos_sim` |
| `--pi_cap` | 0.10 | Upper bound on estimated class prior |
| `--fusion` | gated | Modality fusion: `gated` / `concat` / `sum` |
| `--d_latent` | 256 | Shared latent dimension |
| `--emb_pca_dim` | 256 | PCA dimension for LLM embeddings |
| `--warmup_epochs` | 10 | Stage 1 BCE warmup epochs |
| `--nnpu_epochs` | 25 | Stage 2 nnPU training epochs |

---

## Expected Results

Macro-average Adjusted F1 over 15 tasks and 5 seeds:

| Method | Macro Adjusted F1 |
|--------|------------------|
| **TargetSage** | **11.1%** |
| ResNet | 8.1% |
| Gradient Boosting | 7.6% |
| FT-Transformer | 6.7% |
| Logistic Regression | 6.0% |
| MLP | 6.0% |
| TabNet | 5.1% |
| KNN | 3.7% |
| SVM | 3.3% |
| Naive Bayes | 1.6% |
