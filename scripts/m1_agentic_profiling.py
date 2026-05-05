#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetSage — Module M1: Agentic Profiling
==========================================
NOTE FOR REVIEWERS
-------------------
This script is a documented stub explaining the M1 pipeline.  Precomputed
M1 outputs are provided in data/features_llm_structured_scores.csv and
data/features_llm_embedding.csv, so you DO NOT need to re-run M1 to
reproduce the main results.  Running `python train.py` is sufficient.

This stub is provided so reviewers can understand the full data provenance
pipeline described in Section 3.1 of the paper.

Overview of Module M1: Agentic Profiling
-----------------------------------------
Module M1 constructs a rich, multi-source evidence profile for each of the
19,032 human protein-coding genes using a six-tool agentic pipeline, then
distills the retrieved evidence into 23 structured numerical attributes using
an LLM (Appendix A.1 and A.2 of the paper).

Tool 1: Structured Feature Reader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The agent reads the 482 numerical features aggregated from 15 public databases
(Table 1 of the paper) and converts informative entries into human-readable
evidence statements.  For example, selected entries for ABL1 are rendered as:
  "highly loss-of-function intolerant (pLI=1.00); observed/expected LoF
   ratio: 0.07; essential in mouse; protein length 1130 aa; 43 DGIdb
   interaction types; protein kinase, SH2, and SH3 domains."
This semanticization provides biologically contextualized input for reasoning.

Tools 2–5: External Knowledge Collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The agent queries four external sources to complement the structured features:

    2. NCBI Gene Summary    — gene function, disease context (broad overview)
    3. UniProt Function     — expert-curated molecular function, subcellular
                              localization, and tissue specificity
    4. PubMed Literature    — recent abstracts for mechanistic/therapeutic context
    5. Open Targets Platform — tractability assessments (small molecule, antibody,
                               PROTAC) and known safety liabilities

Across the 19,032-gene corpus, structured features and NCBI summaries are
available for nearly all genes; UniProt annotations and Open Targets
tractability are available for ~85% and ~72% of genes, respectively.
PubMed abstracts are retrieved selectively when additional evidence is needed.

Tool 6: Fact-Checking Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Collecting information from multiple sources introduces two risks:
  (1) Label leakage: external sources may state "FDA-approved drug target"
      directly, leaking the prediction label.
  (2) Inconsistent statements: different databases may describe the same gene
      differently (e.g., conflicting localization or function annotations).
The fact-checker scans collected text for leakage patterns, removes offending
sentences, and flags inconsistent cross-source statements.  In the full corpus,
direct label leakage was detected in 1 of 19,032 genes; cross-source
inconsistencies were found in 385 genes (2.0%).

Step 2: LLM Attribute Scoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The assembled evidence profile is passed to an LLM using PROMPT_TEMPLATE.
The LLM generates:
  (a) A 2-3 sentence biological reasoning chain justifying the scores.
  (b) A JSON dictionary mapping 23 therapeutic attributes to scores in [0, 1].

Step 3: Vocabulary Discovery and Coverage Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Across all 19,032 genes, the agent initially surfaces 2,261 candidate
attribute dimensions.  Attributes appearing in fewer than 5% of gene profiles
are dropped, yielding 28 candidates.  After deduplication of naming variants
(e.g., five RVIS-related names collapsed to "rvis_score"), d_a=23 unique
attributes remain (listed in ATTR_VOCAB_23 below; full details in Appendix A.3).

Step 4: Output
~~~~~~~~~~~~~~~
Two output files are produced per gene set:

    features_llm_structured_scores.csv
        One row per gene, 23 numeric columns — the explicit attribute scores.
        These are fed directly to head_attr in the TargetSage model.

    features_llm_embedding.csv
        One row per gene, 1536 numeric columns — the LLM text embeddings of
        the full evidence profile.  Produced by passing the profile through
        the text-embedding-3-large embedding endpoint.  In train.py, these
        are PCA-compressed to 256 dims before being fed to head_emb.
"""

import os
import re
import json
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 23-attribute therapeutic vocabulary (Appendix A.3 of the paper)
# ---------------------------------------------------------------------------
# These 23 attributes survived the >=5% coverage filter after deduplication.
# The names below are the Python-safe snake_case versions of the names listed
# in Appendix A.3.
ATTR_VOCAB_23 = [
    "disease_association",         # strength of disease-gene links (OMIM, DisGeNET)
    "essential_gene",              # fitness-essential in CRISPR screens / mouse KO
    "loss_of_function_tolerance",  # pLI / LOEUF from gnomAD (lower = more intolerant)
    "loss_of_function_constraint", # supplementary constraint metrics (RVIS, MTR)
    "observed_expected_lof_ratio", # gnomAD observed/expected LoF ratio
    "rvis_score",                  # residual variation intolerance score
    "gwas_associations",           # number / strength of GWAS trait associations
    "protein_protein_interactions",# count of high-confidence PPI partners (STRING)
    "chemical_gene_interactions",  # number of chemical-gene interactions (CTDbase)
    "drug_interaction_types",      # number of DGIdb interaction type categories
    "known_antibodies",            # number of known antibody reagents targeting gene
    "antibody",                    # Open Targets antibody tractability bucket
    "antibody_availability",       # antibody availability for research use
    "small_molecule",              # Open Targets small-molecule tractability bucket
    "tractable_modalities",        # number of tractable drug modalities
    "overall_therapeutic_potential",# aggregate therapeutic potential score
    "functional_characterization", # degree of functional/mechanistic characterization
    "protein_domains",             # number and diversity of InterPro protein domains
    "protein_length",              # protein length in amino acids
    "expression_tissue_specificity",# restricted vs. broad tissue expression (GTEx)
    "expression_specificity",      # tissue-specificity index (tau score)
    "expression_broadly_expressed",# fraction of tissues with detectable expression
    "alternative_splicing",        # number of annotated alternative isoforms
]

# ---------------------------------------------------------------------------
# Prompt template (Section 3.1 / Appendix A.2 of the paper)
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a drug discovery scientist evaluating therapeutic targets.

Given the evidence profile for gene {gene}, write a 2-3 sentence biological
reasoning chain summarizing its therapeutic potential, then score the gene on
the following {n_attrs} attributes on a scale of 0.0 (absent/low) to 1.0 (strong/high):

{attr_list}

Use EXACTLY this output format:
REASONING: <2-3 sentence reasoning chain>
ATTRIBUTES: {{"disease_association": <score>, "essential_gene": <score>, ...}}

Evidence profile for {gene}:
{evidence}
"""

# Few-shot examples (well-characterized targets with known attribute profiles).
FEW_SHOT = [
    {
        "user": (
            "Gene: BRCA1\n"
            "Evidence: Tumor suppressor; DNA repair (homologous recombination); "
            "pLI=1.00; LOEUF=0.07; OMIM: breast/ovarian cancer; "
            "GWAS: breast cancer p=1e-120; STRING: 287 partners; "
            "InterPro: BRCT domain (IPR001357), RING finger; "
            "UniProt: nuclear; Open Targets: small-molecule bucket 2, antibody bucket 2; "
            "1863 aa; GTEx: ubiquitous (tau=0.12); 3 major isoforms."
        ),
        "assistant": (
            "REASONING: BRCA1 is a highly constrained tumor suppressor (pLI=1.00, LOEUF=0.07) "
            "with strong GWAS linkage to breast/ovarian cancer and a well-characterized BRCT "
            "domain supporting DNA repair. Its clinical validation through PARP inhibitor "
            "synthetic lethality (olaparib, rucaparib) makes it a prime small-molecule target.\n"
            'ATTRIBUTES: {"disease_association": 0.98, "essential_gene": 0.80, '
            '"loss_of_function_tolerance": 0.05, "loss_of_function_constraint": 0.08, '
            '"observed_expected_lof_ratio": 0.07, "rvis_score": 0.04, '
            '"gwas_associations": 0.97, "protein_protein_interactions": 0.90, '
            '"chemical_gene_interactions": 0.70, "drug_interaction_types": 0.75, '
            '"known_antibodies": 0.85, "antibody": 0.65, "antibody_availability": 0.80, '
            '"small_molecule": 0.82, "tractable_modalities": 0.80, '
            '"overall_therapeutic_potential": 0.95, "functional_characterization": 0.95, '
            '"protein_domains": 0.80, "protein_length": 0.75, '
            '"expression_tissue_specificity": 0.12, "expression_specificity": 0.15, '
            '"expression_broadly_expressed": 0.95, "alternative_splicing": 0.70}'
        ),
    },
    {
        "user": (
            "Gene: KCNQ1\n"
            "Evidence: Voltage-gated potassium channel; cardiac repolarization; "
            "pLI=0.98; LOEUF=0.11; OMIM: long QT syndrome, familial atrial fibrillation; "
            "GWAS: QT interval p=1e-85, atrial fibrillation p=1e-12; STRING: 54 partners; "
            "InterPro: potassium channel (IPR003968), PAS domain; "
            "UniProt: plasma membrane; Open Targets: small-molecule bucket 1, antibody bucket 1; "
            "676 aa; GTEx: heart-enriched (tau=0.62); 2 major isoforms."
        ),
        "assistant": (
            "REASONING: KCNQ1 is a highly constrained cardiac ion channel (pLI=0.98) with "
            "strong GWAS support for QT prolongation and atrial fibrillation. Its plasma "
            "membrane localization and approved drug interactions (azimilide, chromanol 293B) "
            "confirm small-molecule tractability, while cardiac-restricted expression limits "
            "off-target liability.\n"
            'ATTRIBUTES: {"disease_association": 0.92, "essential_gene": 0.70, '
            '"loss_of_function_tolerance": 0.08, "loss_of_function_constraint": 0.10, '
            '"observed_expected_lof_ratio": 0.11, "rvis_score": 0.06, '
            '"gwas_associations": 0.90, "protein_protein_interactions": 0.50, '
            '"chemical_gene_interactions": 0.60, "drug_interaction_types": 0.65, '
            '"known_antibodies": 0.55, "antibody": 0.40, "antibody_availability": 0.50, '
            '"small_molecule": 0.90, "tractable_modalities": 0.65, '
            '"overall_therapeutic_potential": 0.88, "functional_characterization": 0.88, '
            '"protein_domains": 0.72, "protein_length": 0.50, '
            '"expression_tissue_specificity": 0.62, "expression_specificity": 0.65, '
            '"expression_broadly_expressed": 0.40, "alternative_splicing": 0.45}'
        ),
    },
]

ATTR_LIST_STR = ", ".join(ATTR_VOCAB_23)


# ---------------------------------------------------------------------------
# Tool stubs (Tools 1–6 from Appendix A.2)
# ---------------------------------------------------------------------------

def tool1_read_structured_features(gene_symbol: str, bio_features: dict) -> str:
    """
    Tool 1: Structured Feature Reader.

    Reads the gene's 482 numerical features and converts informative entries
    into human-readable evidence statements, so downstream reasoning operates
    on biologically contextualized evidence rather than raw numbers.

    Parameters
    ----------
    gene_symbol  : HGNC gene symbol
    bio_features : dict mapping feature name to value for this gene

    Returns
    -------
    str  semanticized evidence string (up to 300 characters)
    """
    parts = []
    # Constraint metrics
    pli = bio_features.get("pLI", None)
    if pli is not None:
        intol = "highly" if pli > 0.9 else ("moderately" if pli > 0.5 else "tolerant to")
        parts.append(f"{intol} loss-of-function intolerant (pLI={pli:.2f})")
    oe = bio_features.get("oe_lof_upper", None)
    if oe is not None:
        parts.append(f"observed/expected LoF ratio: {oe:.2f}")
    plen = bio_features.get("protein_length", None)
    if plen:
        parts.append(f"protein length {int(plen)} aa")
    dgi = bio_features.get("n_dgidb_types", None)
    if dgi:
        parts.append(f"{int(dgi)} DGIdb interaction types")
    return "; ".join(parts)[:300]


def tool2_ncbi_gene_summary(gene_symbol: str) -> str:
    """
    Tool 2: NCBI Gene Summary.

    Retrieves the NCBI RefSeq gene summary providing broad functional and
    disease context.  Pre-fetched summaries are available in
    data/gene_summaries.tsv.

    Returns: gene summary string (up to 350 characters).
    API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
    """
    raise NotImplementedError(
        "NCBI gene summaries are precomputed in data/gene_summaries.tsv"
    )


def tool3_uniprot_function(gene_symbol: str) -> str:
    """
    Tool 3: UniProt Function Annotation.

    Retrieves expert-curated molecular function descriptions, subcellular
    localization, and tissue specificity from UniProt.

    Returns: "UniProt: <function>; <localization>; <tissue>"
    API: https://rest.uniprot.org/uniprotkb/search
    """
    raise NotImplementedError("Use precomputed features in data/")


def tool4_pubmed_literature(gene_symbol: str, max_abstracts: int = 3) -> str:
    """
    Tool 4: PubMed Literature.

    Retrieves recent abstracts providing mechanistic or therapeutic context,
    called selectively when additional literature evidence is needed.

    Returns: concatenated abstract snippets (up to 400 characters).
    API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
    """
    raise NotImplementedError("Use precomputed features in data/")


def tool5_open_targets(gene_symbol: str) -> str:
    """
    Tool 5: Open Targets Platform.

    Retrieves tractability assessments (small molecule, antibody, PROTAC)
    and known safety liabilities from Open Targets.

    Returns: "OT: sm_bucket=<n>, ab_bucket=<n>, protac_bucket=<n>"
    API: https://platform.opentargets.org/api
    """
    raise NotImplementedError("Use precomputed features in data/")


def tool6_fact_check(profile: str, label_keywords: Optional[List[str]] = None) -> str:
    """
    Tool 6: Fact-Checking Agent.

    Scans the assembled evidence profile for two types of problems:
      (1) Label leakage: sentences directly stating clinical approval status
          (e.g., "FDA-approved drug target") which would reveal the prediction
          label and must be removed.
      (2) Inconsistent statements: conflicting descriptions across sources
          (e.g., one source says "nuclear", another says "cytoplasmic").

    Parameters
    ----------
    profile         : assembled multi-source evidence profile string
    label_keywords  : additional keywords to treat as leakage signals

    Returns
    -------
    str  cleaned evidence profile with leakage sentences removed
    """
    leakage_patterns = [
        r"FDA[\-\s]approved", r"EMA[\-\s]approved", r"approved drug target",
        r"clinically approved", r"approved for treatment",
    ]
    if label_keywords:
        leakage_patterns += [re.escape(k) for k in label_keywords]

    sentences = re.split(r'(?<=[.!?])\s+', profile)
    clean = []
    for sent in sentences:
        if any(re.search(p, sent, re.IGNORECASE) for p in leakage_patterns):
            continue  # remove leakage sentence
        clean.append(sent)
    return " ".join(clean)


# ---------------------------------------------------------------------------
# Evidence aggregation
# ---------------------------------------------------------------------------

def build_evidence_profile(
    gene: str,
    tool1_bio: str,
    tool2_summary: str,
    tool3_uniprot: str,
    tool4_pubmed: str,
    tool5_ot: str,
    max_len: int = 600,
) -> str:
    """
    Concatenate evidence from Tools 1–5 into a single profile string,
    then apply fact-checking (Tool 6).

    The profile is truncated to max_len characters.
    """
    parts = [p for p in [tool2_summary, tool1_bio, tool3_uniprot,
                         tool5_ot, tool4_pubmed] if p]
    raw = " | ".join(parts)
    cleaned = tool6_fact_check(raw)
    return (cleaned[:max_len] + "...") if len(cleaned) > max_len else cleaned


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def parse_llm_response(text: str) -> tuple:
    """
    Extract structured attributes and reasoning from LLM output.

    Returns
    -------
    attrs     : dict mapping attribute name to float score in [0, 1]
    reasoning : str  biological reasoning chain (up to 500 chars)
    """
    attrs = {}

    m = re.search(r'ATTRIBUTES:\s*(\{[^{}]*\})', text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            for k, v in obj.items():
                if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
                    attrs[k] = float(v)
        except (json.JSONDecodeError, ValueError):
            pass

    if len(attrs) < 5:
        for key in ATTR_VOCAB_23:
            pat = rf'["\s]*{re.escape(key)}["\s]*[:=]\s*([0-9]*\.?[0-9]+)'
            m2 = re.search(pat, text, re.IGNORECASE)
            if m2:
                try:
                    v = float(m2.group(1))
                    if 0.0 <= v <= 1.0:
                        attrs[key] = v
                except ValueError:
                    pass

    reasoning = ""
    rm = re.search(r'REASONING:\s*(.*?)(?:ATTRIBUTES:|\{|$)', text, re.DOTALL | re.IGNORECASE)
    if rm:
        reasoning = rm.group(1).strip()[:500]

    return attrs, reasoning


def attrs_to_vector(attrs: dict, vocab: List[str]) -> np.ndarray:
    """
    Convert attribute dict to a fixed-length vector.
    Missing attributes default to 0.5 (neutral / uncertain).
    """
    return np.array([attrs.get(k, 0.5) for k in vocab], dtype=np.float32)


# ---------------------------------------------------------------------------
# Coverage filtering
# ---------------------------------------------------------------------------

def filter_by_coverage(
    attr_matrix: pd.DataFrame,
    min_coverage: float = 0.05,
) -> pd.DataFrame:
    """
    Drop attribute columns present in fewer than min_coverage fraction of genes.

    In the paper, this step reduced 2,261 initial candidate attributes to
    28, which after deduplication yielded the final 23-attribute vocabulary.
    """
    n_genes = len(attr_matrix)
    valid_cols = [
        c for c in attr_matrix.columns
        if attr_matrix[c].notna().sum() / n_genes >= min_coverage
    ]
    dropped = set(attr_matrix.columns) - set(valid_cols)
    if dropped:
        print(f"[M1] Dropped {len(dropped)} low-coverage attributes: {sorted(dropped)}")
    return attr_matrix[valid_cols]


# ---------------------------------------------------------------------------
# Main pipeline stub
# ---------------------------------------------------------------------------

def run_m1_pipeline(
    gene_list: List[str],
    output_scores_path: str,
    output_embeddings_path: str,
    llm_client=None,
    embedding_client=None,
):
    """
    Full M1 pipeline stub.

    For each gene in gene_list:
      1. Tool 1: Semanticize structured bio features.
      2. Tool 2: Retrieve NCBI Gene Summary.
      3. Tool 3: Retrieve UniProt function annotation.
      4. Tool 4: Retrieve PubMed literature abstracts (selective).
      5. Tool 5: Retrieve Open Targets tractability scores.
      6. Tool 6: Fact-check and assemble the evidence profile.
      7. Call LLM with PROMPT_TEMPLATE to get 23-attribute vector + reasoning.
      8. Obtain 1536-dim text embedding via text-embedding-3-large.
    After all genes:
      9. Apply coverage filter (>=5% genes must have each attribute).
      10. Save 23-attribute matrix to output_scores_path.
      11. Save 1536-dim embeddings to output_embeddings_path.

    NOTE: This function raises NotImplementedError because the precomputed
    outputs are already provided in data/.
    """
    raise NotImplementedError(
        "\n"
        "Module M1 (Agentic Profiling) requires:\n"
        "  - API access for: NCBI Gene, UniProt, PubMed, Open Targets\n"
        "  - An LLM API key for attribute scoring (text generation)\n"
        "  - An embedding API key (text-embedding-3-large)\n"
        "  - ~48-72 hours of API calls for all 19,032 human genes\n"
        "\n"
        "Precomputed M1 outputs are already provided in the data/ directory:\n"
        "  data/features_llm_structured_scores.csv   (23-dim attribute scores)\n"
        "  data/features_llm_embedding.csv           (1536-dim LLM embeddings)\n"
        "\n"
        "To reproduce the main results, run:\n"
        "  python train.py\n"
    )


if __name__ == "__main__":
    print(__doc__)
    print("=" * 70)
    print("Module M1 is a documented stub.")
    print("Precomputed outputs are in data/features_llm_structured_scores.csv")
    print("and data/features_llm_embedding.csv")
    print("")
    print("To reproduce main results, run: python train.py")
    print("=" * 70)
