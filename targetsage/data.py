"""
Data Loading Utilities for TargetSage / TargetSage
===================================================
This module provides loaders for the four data files used by TargetSage and
helper functions to merge all modalities into a single aligned feature matrix.

Expected Data Files
--------------------
All files should be placed in the data/ directory relative to the working
directory (or paths can be overridden via CLI arguments in train.py):

    gene_features.tsv
        TSV, ~19 032 rows × 483 columns (Gene_Symbol + 482 feature columns).
        Contains structured biological features assembled from public databases:
        - Constraint metrics (pLI, LOEUF, missense z-score from gnomAD)
        - Tissue expression (GTEx TPM across 54 tissues)
        - Protein–protein interaction network centrality (STRING)
        - Pathway membership (Reactome, KEGG)
        - GWAS association counts, OMIM disease links
        - Target class annotations (kinase, GPCR, ion channel, etc.)
        These features are task-agnostic and apply to all 15 benchmark tasks.

    features_llm_structured_scores.csv
        CSV, ~19 032 rows × 24 columns (Gene_Symbol + 23 attribute columns).
        Contains the 23 LLM-derived explicit attribute scores produced by
        Module M1 (Agentic Profiling) and refined by Module M2 (GRPO).
        Each value is a float in [0, 1] representing the model's confidence
        that the gene has that therapeutic attribute (e.g., druggability,
        cancer_relevance, tractable_small_molecule, etc.).

    features_llm_embedding.csv
        CSV, ~19 032 rows × 1537 columns (Gene_Symbol + 1536 embedding dims).
        Contains raw LLM text embeddings of per-gene evidence summaries.
        In train.py these are compressed to 256 dims via PCA before being
        fed to head_emb.  The raw 1536-dim vectors capture semantic similarity
        between genes based on their biological evidence profiles.

    gene_labels.tsv  (also referred to as gene_labels_enriched.tsv)
        TSV, ~19 032 rows × 16 columns (Gene_Symbol + 15 binary task columns).
        Binary labels (0/1) for each of the 15 benchmark tasks.  A value of 1
        means the gene is a known positive for that task; 0 means unlabeled
        (could be a true negative OR an undiscovered positive).

    prior_task_summary.csv
        CSV with columns [task, pi_median, pi_mean, pi_std].
        Contains LLM-estimated class priors π_llm for each task, obtained by
        asking an LLM to estimate the fraction of human genes that satisfy
        each task definition.  Used in the hybrid prior π = α·π_data + (1-α)·π_llm.

Column Prefix Scheme
---------------------
After `build_feature_matrix()` merges the four DataFrames on Gene_Symbol,
non-key columns are prefixed to avoid name collisions:

    bio__<col>   — columns from gene_features.tsv
    attr__<col>  — columns from features_llm_structured_scores.csv
    emb__<col>   — columns from features_llm_embedding.csv

`get_feature_arrays()` then recovers these blocks by prefix, producing
numpy arrays X_bio, X_attr, X_emb ready for the TargetSage model.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# The 15 benchmark tasks, ordered as they appear in the paper Table 2.
# Each key is a column name in gene_labels.tsv.
TASKS = [
    "task_pharos_tclin_vs_others",
    "task_pharos_tclin_tchem_vs_others",
    "task_triage_tier1_vs_others",
    "task_triage_tier12_vs_others",
    "task_cancer_druggability",
    "task_ab_bucket1_vs_others",
    "task_ab_bucket123_vs_others",
    "task_sm_bucket1_vs_others",
    "task_sm_bucket123_vs_others",
    "task_protac_bucket1234_vs_others",
    "task_cancer_type_specific_target_prioritization",
    "task_pan_cancer_target_prioritization",
    "task_T1_targets_only",
    "task_T1_T2_targets",
    "task_T1_T2_T3_targets",
]

# Human-readable display names used in result tables and plots.
# Keys match TASKS list above.
TASK_DISPLAY = {
    "task_pharos_tclin_vs_others":                     "Clinical Targets",
    "task_pharos_tclin_tchem_vs_others":               "Clinical & Chemical",
    "task_triage_tier1_vs_others":                     "Top-Tier Targets",
    "task_triage_tier12_vs_others":                    "High-Confidence Targets",
    "task_cancer_druggability":                        "Cancer-Relevant",
    "task_cancer_type_specific_target_prioritization": "Cancer Type-Specific",
    "task_pan_cancer_target_prioritization":           "Pan-Cancer",
    "task_T1_targets_only":                            "T1 Cancer",
    "task_T1_T2_targets":                              "T1-T2 Cancer",
    "task_T1_T2_T3_targets":                           "T1-T3 Cancer",
    "task_sm_bucket1_vs_others":                       "Small-Molecule (Approved)",
    "task_sm_bucket123_vs_others":                     "Small-Molecule (Clinical+)",
    "task_ab_bucket1_vs_others":                       "Antibody (Approved)",
    "task_ab_bucket123_vs_others":                     "Antibody (Clinical+)",
    "task_protac_bucket1234_vs_others":                "PROTAC Targets",
}

# Reverse map: display name → task key.  Used to look up π_llm from the
# prior_task_summary.csv which indexes by display name.
DISPLAY_TO_TASK = {v: k for k, v in TASK_DISPLAY.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Candidate column names for the gene identifier column across different files.
# The loader normalizes all of these to "Gene_Symbol".
_GENE_COL_CANDIDATES = [
    "Gene_Symbol", "gene_symbol", "gene", "Gene",
    "symbol", "Symbol", "HGNC", "hgnc_symbol",
]


def _fix_gene_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the gene identifier column is named 'Gene_Symbol'.

    Different source files use different column names for the gene identifier.
    This function checks a list of known candidates and renames the matching
    column.  If no candidate is found, the first column is assumed to be the
    gene identifier (handles files with an unnamed index column from pandas).
    """
    if "Gene_Symbol" in df.columns:
        return df  # Already in canonical form
    for c in _GENE_COL_CANDIDATES:
        if c in df.columns:
            return df.rename(columns={c: "Gene_Symbol"})
    # Fall back: pandas sometimes writes index as "Unnamed: 0"
    if "Unnamed: 0" in df.columns:
        return df.rename(columns={"Unnamed: 0": "Gene_Symbol"})
    # Last resort: treat the first column as gene symbols
    return df.rename(columns={df.columns[0]: "Gene_Symbol"})


def _dedup_gene(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate gene rows, keeping the first occurrence.

    Duplicate Gene_Symbol entries can arise from database merges or
    alternative gene name aliases.  We keep the first occurrence, which
    is typically the canonical HGNC symbol.
    """
    if "Gene_Symbol" in df.columns and df["Gene_Symbol"].duplicated().any():
        df = df.drop_duplicates(subset=["Gene_Symbol"], keep="first")
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_bio_features(path: str) -> pd.DataFrame:
    """
    Load the structured biological feature matrix.

    Reads gene_features.tsv (tab-separated, ~19 032 genes × 483 columns).
    Returns a DataFrame with 'Gene_Symbol' as the first column and 482
    numeric feature columns.  Duplicates are dropped.

    Parameters
    ----------
    path : str  path to gene_features.tsv

    Returns
    -------
    pd.DataFrame with columns ['Gene_Symbol', feat_1, feat_2, ...]
    """
    df = pd.read_csv(path, sep="\t")
    return _dedup_gene(_fix_gene_col(df))


def load_llm_scores(path: str) -> pd.DataFrame:
    """
    Load LLM-derived explicit attribute scores (M1/M2 output).

    Reads features_llm_structured_scores.csv (comma-separated).
    Each row is a gene; each column (after Gene_Symbol) is one of the
    23 therapeutic attributes scored in [0, 1].  These scores were
    produced by Module M1 (agentic profiling) and optionally refined
    by Module M2 (GRPO fine-tuning).

    Parameters
    ----------
    path : str  path to features_llm_structured_scores.csv

    Returns
    -------
    pd.DataFrame with columns ['Gene_Symbol', attr_1, ..., attr_23]
    """
    df = pd.read_csv(path)
    return _dedup_gene(_fix_gene_col(df))


def load_llm_embeddings(path: str) -> pd.DataFrame:
    """
    Load LLM text embedding features.

    Reads features_llm_embedding.csv (comma-separated).
    Each row is a gene; columns are the 1536 embedding dimensions from the
    LLM encoder (e.g., text-embedding-3-large).  In train.py
    these are further compressed to 256 dims by PCA.

    Parameters
    ----------
    path : str  path to features_llm_embedding.csv

    Returns
    -------
    pd.DataFrame with columns ['Gene_Symbol', dim_0, ..., dim_1535]
    """
    df = pd.read_csv(path)
    return _dedup_gene(_fix_gene_col(df))


def load_labels(path: str) -> pd.DataFrame:
    """
    Load binary task labels for all 15 benchmark tasks.

    Reads gene_labels.tsv (tab-separated).  Returns a DataFrame containing
    only the 'Gene_Symbol' column and the 15 task columns defined in TASKS.
    Raises ValueError if any expected task column is missing.

    Parameters
    ----------
    path : str  path to gene_labels.tsv

    Returns
    -------
    pd.DataFrame with columns ['Gene_Symbol', task_1, ..., task_15]
    """
    df  = pd.read_csv(path, sep="\t")
    df  = _fix_gene_col(df)
    # Verify that all 15 task columns are present
    needed  = ["Gene_Symbol"] + TASKS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in {path}: {missing}")
    return _dedup_gene(df[needed].copy())


def load_prior_map(path: str) -> Dict[str, float]:
    """
    Load the LLM-estimated class prior π per task.

    Reads prior_task_summary.csv which must have columns ['task', 'pi_median'].
    The 'task' column uses display names (values of TASK_DISPLAY), not task keys.
    Returns a dict mapping display_name → π_median.

    This dict is used in train.py to compute the hybrid prior:
        π = α · π_data + (1-α) · π_llm

    Parameters
    ----------
    path : str  path to prior_task_summary.csv

    Returns
    -------
    dict mapping task display name (str) to median LLM prior estimate (float)
    """
    df = pd.read_csv(path)
    return {str(r["task"]): float(r["pi_median"]) for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    bio_df: pd.DataFrame,
    score_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all modalities into a single aligned DataFrame on Gene_Symbol.

    Uses inner joins so only genes present in ALL four DataFrames are
    retained (typically ~17 000–18 000 genes after the inner join over
    four sources, depending on LLM coverage).

    Column prefix scheme
    --------------------
    To avoid name collisions when columns from different files share the
    same name, non-key columns are prefixed before merging:

        bio__<col>   columns from bio_df (gene_features.tsv)
        attr__<col>  columns from score_df (features_llm_structured_scores.csv)
        emb__<col>   columns from emb_df (features_llm_embedding.csv)

    Task label columns (task_*) and Gene_Symbol are NOT prefixed.

    Parameters
    ----------
    bio_df    : output of load_bio_features()
    score_df  : output of load_llm_scores()
    emb_df    : output of load_llm_embeddings()
    labels_df : output of load_labels()

    Returns
    -------
    pd.DataFrame with columns:
        ['Gene_Symbol', 'task_*' × 15, 'bio__*' × 482,
         'attr__*' × 23, 'emb__*' × 1536]
    """
    def _prefix(df, prefix):
        # Add prefix to every column except the gene identifier
        return df.rename(columns={c: f"{prefix}{c}"
                                  for c in df.columns if c != "Gene_Symbol"})

    # Start with labels (defines the gene universe for the merge)
    merged = labels_df.copy()
    merged = merged.merge(_prefix(bio_df,   "bio__"),  on="Gene_Symbol", how="inner")
    merged = merged.merge(_prefix(score_df, "attr__"), on="Gene_Symbol", how="inner")
    merged = merged.merge(_prefix(emb_df,   "emb__"),  on="Gene_Symbol", how="inner")
    return merged


def get_feature_arrays(merged: pd.DataFrame):
    """
    Split the merged DataFrame into separate numpy arrays per modality.

    Uses the column prefix scheme set by build_feature_matrix() to identify
    which columns belong to each modality.

    Parameters
    ----------
    merged : output of build_feature_matrix()

    Returns
    -------
    X_bio   : float32 numpy array [N, 482]   — structured bio features
    X_attr  : float32 numpy array [N, 23]    — LLM attribute scores
    X_emb   : float32 numpy array [N, 1536]  — LLM embeddings (pre-PCA)
    genes   : numpy array [N]  — gene symbol strings in the same row order
    """
    bio_cols  = [c for c in merged.columns if c.startswith("bio__")]
    attr_cols = [c for c in merged.columns if c.startswith("attr__")]
    emb_cols  = [c for c in merged.columns if c.startswith("emb__")]

    Xb    = merged[bio_cols].to_numpy(dtype=np.float32)
    Xa    = merged[attr_cols].to_numpy(dtype=np.float32)
    Xe    = merged[emb_cols].to_numpy(dtype=np.float32)
    genes = merged["Gene_Symbol"].to_numpy()

    return Xb, Xa, Xe, genes
