# TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning

Reviewer package for NeurIPS 2026 submission.

---

## Overview

TargetSage is a three-module framework for prioritizing human protein-coding genes as drug targets across 19,032 genes and 15 benchmark tasks. Module M1 (Agentic Profiling) queries six biomedical knowledge bases per gene and distills evidence into structured LLM attribute scores. Module M2 (Guided Reasoning, GRPO) fine-tunes the attribute-scoring LLM using reinforcement learning so that generated attributes complement structural features. Module M3 (PU-Aware Scoring) trains TargetSage, a three-head gated-fusion MLP discriminator, using the nnPU loss with a hybrid class prior and semantic reweighting. TargetSage achieves a macro-average Adjusted F1 of 11.1% across 15 tasks versus 8.1% for the best deep tabular baseline (ResNet) and 7.6% for the best classical baseline (Gradient Boosting).

---

## Repository Structure

```
targetsage_review_package/
├── README.md                        This file
├── requirements.txt                 Python dependencies
├── train.py                         M3: train TargetSage, evaluate on 15 tasks
├── inference.py                     M3: full-genome ranking (no held-out split)
├── targetsage/
│   ├── __init__.py                  Public API
│   ├── model.py                     TargetSage architecture (3-head gated fusion)
│   ├── loss.py                      nnPU loss (Kiryo 2017) with semantic weights
│   ├── metrics.py                   Adjusted F1 = R_soft^2 / p_bar
│   └── data.py                      Data loaders and feature matrix builder
└── scripts/
    ├── m1_agentic_profiling.py      M1 documented stub + prompt template
    ├── m2_grpo_training.py          M2 GRPO fine-tuning pipeline
    └── run_baselines.py             All 10 baselines: classical ML + ResNet/TabNet/FT-Trans
```

Data files (placed in `data/` before running):
```
data/
├── gene_features.tsv                ~19032 genes x 482 structured bio features
├── features_llm_structured_scores.csv  ~19032 genes x 23 LLM attribute scores (M1/M2 output)
├── features_llm_embedding.csv       ~19032 genes x 1536 LLM text embeddings (M1 output)
├── gene_labels.tsv                  ~19032 genes x 15 binary task labels
└── prior_task_summary.csv           LLM-estimated class prior pi per task
```

---

## Quick Start

The `data/` directory with precomputed M1 and M2 outputs must be present. All M1 (agentic profiling) and M2 (GRPO) outputs are precomputed; reviewers only need to run M3 training.

```bash
pip install -r requirements.txt

# Quick test: 2 tasks, 1 seed (~5 min on CPU, ~1 min on GPU)
python train.py --seeds 0,1,2 --tasks task_pharos_tclin_vs_others,task_T1_targets_only
```

---

## Full 15-Task Reproduction

```bash
# All 15 tasks, 5 seeds — approximately 3 hours on RTX 3090
python train.py
```

Results are saved to `results/train_<timestamp>/`:
- `results.csv` — per-task mean ± std Adjusted F1 over 5 seeds
- `raw.csv` — per-task per-seed rows
- `config.json` — full run configuration

---

## Baseline Reproduction

```bash
python scripts/run_baselines.py
```

Results are saved to `results/baselines/classical_baselines_results.csv` and `results/baselines/deep_baselines_results.csv`.

```bash
# Classical ML only (fast)
python scripts/run_baselines.py --mode classical

# Deep tabular only (ResNet, TabNet, FT-Transformer)
python scripts/run_baselines.py --mode deep

# Quick test with fewer seeds
python scripts/run_baselines.py --seeds 0 1 2
```

---

## Data Format

| File | Format | Columns | Description |
|------|--------|---------|-------------|
| `gene_features.tsv` | TSV | Gene_Symbol + 482 numeric | Structured bio features: constraint metrics (pLI, LOEUF), tissue expression (GTEx, 54 tissues), PPI network centrality, pathway membership, GWAS counts, OMIM links, target class annotations |
| `features_llm_structured_scores.csv` | CSV | Gene_Symbol + 23 numeric [0,1] | LLM-derived attribute scores for 23 therapeutic properties (disease_association, essential_gene, loss_of_function_tolerance, etc.; full list in Appendix A.3) produced by M1 and refined by M2 |
| `features_llm_embedding.csv` | CSV | Gene_Symbol + 1536 numeric | Raw LLM text embeddings of per-gene evidence summaries; compressed to 256 dims by PCA in train.py before use in head_emb |
| `gene_labels.tsv` | TSV | Gene_Symbol + 15 binary task columns | Binary labels: 1 = known positive target, 0 = unlabeled (may include undiscovered positives). Task names correspond to Pharos, Triage, Cancer, Antibody, Small-Molecule, and PROTAC target sets |
| `prior_task_summary.csv` | CSV | task, pi_median, pi_mean, pi_std | LLM-estimated class prior (fraction of genes that satisfy each task definition); used in hybrid prior pi = alpha * pi_data + (1-alpha) * pi_llm |

---

## Module Descriptions

**Module M1 — Agentic Profiling** (`scripts/m1_agentic_profiling.py`): For each of the 19,032 human protein-coding genes, a six-tool agentic pipeline builds an enriched evidence profile: Tool 1 (Structured Feature Reader) semanticizes the 482 numerical bio features into human-readable statements; Tools 2–5 retrieve complementary evidence from NCBI Gene summaries, UniProt functional annotations, PubMed literature, and Open Targets tractability scores; Tool 6 (Fact-Checking Agent) removes label-leaking sentences and flags cross-source inconsistencies. An LLM then scores each gene on an initial set of ~40 candidate attributes (see `PROMPT_TEMPLATE` in the stub). Across 19,032 genes, 2,261 candidate attributes are surfaced; after coverage filtering (≥5%) and deduplication, the final 23-attribute vocabulary is retained (Appendix A.3). Two output files are produced: `features_llm_structured_scores.csv` (explicit attribute scores) and `features_llm_embedding.csv` (1536-dim embeddings via text-embedding-3-large). Precomputed outputs are provided in `data/`; reviewers do not need to re-run M1.

**Module M2 — Guided Reasoning via GRPO** (`scripts/m2_grpo_training.py`): M2 fine-tunes a local instruction-following LLM (Qwen2.5-1.5B-Instruct with LoRA r=32) using Group Relative Policy Optimization (GRPO) to generate attribute scores that complement the structured bio features for downstream target classification. The reward signal is the macro-average Adjusted F1 of a frozen TargetSage (M3) proxy pre-trained on a ~1000-gene labeled subset: for each GRPO rollout the generated explicit attribute vector is substituted into the attribute input of M3, and the mean-pooled last-layer hidden states of the generated reasoning trace are projected (fixed W_proj) into the M3 embedding input, so that both the explicit scores and the depth of reasoning jointly shape the reward. Within each GRPO step, G=4 rollouts are generated per gene; the group-relative advantage normalizes for per-gene difficulty. The policy gradient update uses standard language model cross-entropy loss scaled by the normalized advantage. Precomputed M2 outputs are provided in `data/features_llm_structured_scores.csv`; reviewers do not need to re-run M2.

**Module M3 — PU-Aware Scoring** (`train.py`, `targetsage/`): M3 trains TargetSage, a three-head gated-fusion MLP, using the nnPU loss with a hybrid class prior and semantic reweighting. Three independent MLP heads encode bio features (482-dim), LLM attribute scores (23-dim), and LLM embeddings (256-dim after PCA) into a shared 256-dim latent space. A gated fusion layer combines the three representations using a learned softmax-weighted sum. Training proceeds in two stages: (1) a 10-epoch BCE warmup to initialize the model, followed by (2) 25 epochs of nnPU training with per-sample unlabeled weights w_j = beta * D(z_j) + (1-beta) * cos(h_e_j, centroid_pos). The class prior pi used in nnPU is a convex combination of a data-driven Elkan-Noto estimate and the LLM prior from M1.

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.6 | Hybrid prior weight: pi = alpha * pi_data + (1-alpha) * pi_llm |
| `beta` | 0.6 | Semantic reweighting: w = beta * D(z) + (1-beta) * cos_sim |
| `d_latent` | 256 | Shared latent dimension for all three MLP heads |
| `head_h` | 512 | Hidden layer width inside each MLPHead |
| `warmup_epochs` | 10 | BCE warmup epochs (Stage 1) |
| `nnpu_epochs` | 25 | nnPU training epochs (Stage 2) |
| `lr` | 2e-4 | AdamW learning rate for both stages |
| `emb_pca_dim` | 256 | PCA output dimension for LLM embeddings (from 1536) |
| `pi_cap` | 0.10 | Maximum allowed class prior (prevents over-estimation) |
| `dropout` | 0.2 | Dropout rate in MLPHead layers |
| `batch_size` | 512 | Mini-batch size for both stages |
| `fusion` | gated | Fusion strategy (gated / concat / sum) |

---

## Expected Results

Macro-average Adjusted F1 over 15 tasks and 5 random seeds (Table 1 of the paper):

| Method | Macro Adjusted F1 |
|--------|------------------|
| TargetSage (this code) | **11.1** |
| ResNet (best deep tabular) | 8.1 |
| Gradient Boosting (best classical) | 7.6 |
| FT-Transformer | 6.7 |
| Logistic Regression | 6.0 |
| MLP (sklearn) | 6.0 |
| TabNet | 5.1 |
| SVM | 3.3 |
| KNN | 3.7 |
| Naive Bayes | 1.6 |

Variance across seeds is typically ±0.3–1.2% depending on the task.  Tasks with very few known positives (e.g., T1 Cancer targets, |P|<100) show higher variance.

---

## Notes on M1 and M2 Precomputed Outputs

Module M1 (agentic profiling) requires API keys for six biomedical databases plus an LLM API key, and approximately 48-72 hours of API calls for all 19,032 genes. Module M2 (GRPO training) requires a GPU with at least 16GB VRAM and approximately 8-12 hours per task. Both M1 and M2 outputs are provided precomputed in the `data/` directory. Reviewers can reproduce the paper's main results (Table 2) by running `python train.py` alone, which uses these precomputed features as input to Module M3.
