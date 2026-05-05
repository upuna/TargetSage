#!/usr/bin/env python3
"""
Agentic Feature Understanding & Enrichment Pipeline

Stage 1 (Static Agent): Convert numerical gene_features.tsv into semantic text profiles
Stage 2 (Dynamic Agent): Collect additional text descriptions from external sources
Stage 3: LLM processes all text into structured attribute vectors

Usage:
  python -u scripts/agent_feature_pipeline.py --stage static   # Numerical → Text
  python -u scripts/agent_feature_pipeline.py --stage dynamic  # External enrichment
  python -u scripts/agent_feature_pipeline.py --stage extract  # Text → Attributes
  python -u scripts/agent_feature_pipeline.py --stage all
"""

import os, sys, json, argparse, time
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── Feature semantics mapping ────────────────────────────────────────────────
# Human-readable descriptions for each feature column
FEATURE_SEMANTICS = {
    # ExAC / Genomic
    "GeneSize": ("Gene size", "bp"),
    "ExAC_cds_len": ("Coding sequence length", "bp"),
    "ExAC_gene_length": ("Genomic span", "bp"),
    "ExAC_cnv.score": ("Copy number variation score", ""),
    "ExAC_flag": ("ExAC quality flag", ""),
    # GnomAD constraint
    "GnomAD_pLI": ("Probability of loss-of-function intolerance (pLI)", "0-1, higher=more constrained"),
    "GnomAD_oe_lof": ("Observed/expected loss-of-function ratio", "lower=more constrained"),
    "GnomAD_oe_mis": ("Observed/expected missense ratio", ""),
    "GnomAD_lof_z": ("Loss-of-function Z-score", "higher=more constrained"),
    "GnomAD_mis_z": ("Missense Z-score", ""),
    "GnomAD_obs_lof": ("Observed loss-of-function variants", "count"),
    "GnomAD_exp_lof": ("Expected loss-of-function variants", "count"),
    # Genic intolerance
    "GenicIntolerance_RVIS": ("Residual Variation Intolerance Score", "lower=more intolerant"),
    "GenicIntolerance_RVIS_ExAC": ("RVIS from ExAC", ""),
    "GenicIntolerance_MTR_ExACv2": ("Missense Tolerance Ratio", "lower=less tolerant"),
    # Mouse
    "MouseGenes_Essential": ("Mouse ortholog is essential", "0/1"),
    "MouseGenes_Non-essential": ("Mouse ortholog is non-essential", "0/1"),
    "MouseGenes_Overexpressed_in_the_brain": ("Overexpressed in brain (mouse)", "0/1"),
    # GWAS
    "GWAS_hits": ("Number of GWAS associations", "count"),
    "GWAS_max_OR": ("Maximum GWAS odds ratio", ""),
    "GWAS_min_P_VALUE": ("Most significant GWAS p-value", ""),
    # Disease
    "MGI_essential_gene": ("Essential gene in mouse", "0/1"),
    "OMIM_uniq_diseases": ("Number of OMIM disease associations", "count"),
    # Druggability
    "DGIdb_interaction_types": ("Number of drug-gene interaction types (DGIdb)", "count"),
    "antibodyCount": ("Number of known antibodies", "count"),
    "monoclonalCount": ("Number of monoclonal antibodies", "count"),
    # Network
    "ppiCount": ("Number of protein-protein interactions", "count"),
    "uniprot_seq_len": ("Protein sequence length (UniProt)", "aa"),
    # Expression
    "GTEx_spec": ("GTEx tissue expression specificity", "higher=more tissue-specific"),
    "hpa_RNA_spec": ("HPA RNA expression specificity", ""),
    "hpa_prot_spec": ("HPA protein expression specificity", ""),
    # STRING
    "string_db_L1_protein_seed_genes_overlap": ("STRING L1 protein network overlap", ""),
    "string_db_L2_protein_seed_genes_overlap": ("STRING L2 protein network overlap", ""),
}

# CTDbase interaction types
CTD_PREFIXES = {
    "ctd_affects^abundance": "affects protein abundance",
    "ctd_affects^activity": "affects enzymatic activity",
    "ctd_affects^binding": "affects molecular binding",
    "ctd_affects^expression": "affects gene expression",
    "ctd_affects^folding": "affects protein folding",
    "ctd_affects^localization": "affects subcellular localization",
    "ctd_affects^metabolic processing": "affects metabolic processing",
    "ctd_affects^phosphorylation": "affects phosphorylation",
    "ctd_affects^splicing": "affects RNA splicing",
    "ctd_affects^transport": "affects molecular transport",
    "ctd_increases^expression": "increases gene expression",
    "ctd_decreases^expression": "decreases gene expression",
    "ctd_increases^activity": "increases enzymatic activity",
    "ctd_decreases^activity": "decreases enzymatic activity",
}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Static Agent - Convert numerical features to text
# ══════════════════════════════════════════════════════════════════════════════
def stage_static(args):
    """Convert gene_features.tsv numerical values into semantic text profiles."""
    print("[STATIC AGENT] Loading gene features...")
    df = pd.read_csv(args.bio_features, sep="\t")
    genes = df["Gene_Symbol"].astype(str).tolist()
    cols = [c for c in df.columns if c != "Gene_Symbol"]
    print(f"  {len(genes)} genes, {len(cols)} features")

    outpath = os.path.join(args.outdir, "gene_text_profiles.jsonl")
    os.makedirs(args.outdir, exist_ok=True)

    # Resume
    existing = set()
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                existing.add(json.loads(line)["gene"])
        print(f"  Resuming: {len(existing)} already done")

    todo = [g for g in genes if g not in existing]
    print(f"  Generating text profiles for {len(todo)} genes...")

    with open(outpath, "a") as fout:
        for idx, gene in enumerate(todo):
            row = df[df["Gene_Symbol"] == gene].iloc[0]
            sections = []

            # ── Genomic constraint ──
            genomic = []
            pli = row.get("GnomAD_pLI", np.nan)
            if pd.notna(pli):
                if pli > 0.9:
                    genomic.append(f"highly loss-of-function intolerant (pLI={pli:.2f})")
                elif pli > 0.5:
                    genomic.append(f"moderately LoF intolerant (pLI={pli:.2f})")
                else:
                    genomic.append(f"LoF tolerant (pLI={pli:.2f})")

            oe_lof = row.get("GnomAD_oe_lof", np.nan)
            if pd.notna(oe_lof):
                genomic.append(f"observed/expected LoF ratio={oe_lof:.2f}")

            rvis = row.get("GenicIntolerance_RVIS", np.nan)
            if pd.notna(rvis):
                genomic.append(f"RVIS={rvis:.3f}")

            essential = row.get("MouseGenes_Essential", np.nan)
            if pd.notna(essential) and essential > 0:
                genomic.append("essential gene in mouse")

            gene_size = row.get("GeneSize", np.nan)
            if pd.notna(gene_size):
                genomic.append(f"gene size={int(gene_size)} bp")

            seq_len = row.get("uniprot_seq_len", np.nan)
            if pd.notna(seq_len) and seq_len > 0:
                genomic.append(f"protein length={int(seq_len)} aa")

            if genomic:
                sections.append(("Genomic constraint", "; ".join(genomic)))

            # ── Disease associations ──
            disease = []
            gwas_hits = row.get("GWAS_hits", np.nan)
            if pd.notna(gwas_hits) and gwas_hits > 0:
                disease.append(f"{int(gwas_hits)} GWAS associations")
                min_p = row.get("GWAS_min_P_VALUE", np.nan)
                if pd.notna(min_p) and min_p > 0:
                    disease.append(f"best p-value={min_p:.2e}")

            omim = row.get("OMIM_uniq_diseases", np.nan)
            if pd.notna(omim) and omim > 0:
                disease.append(f"{int(omim)} OMIM disease(s)")

            mgi = row.get("MGI_essential_gene", np.nan)
            if pd.notna(mgi) and mgi > 0:
                disease.append("MGI essential gene")

            if disease:
                sections.append(("Disease associations", "; ".join(disease)))

            # ── Druggability ──
            drug = []
            dgidb = row.get("DGIdb_interaction_types", np.nan)
            if pd.notna(dgidb) and dgidb > 0:
                drug.append(f"{int(dgidb)} drug interaction type(s) in DGIdb")

            ab_count = row.get("antibodyCount", np.nan)
            if pd.notna(ab_count) and ab_count > 0:
                drug.append(f"{int(ab_count)} known antibodies")

            mono = row.get("monoclonalCount", np.nan)
            if pd.notna(mono) and mono > 0:
                drug.append(f"{int(mono)} monoclonal antibodies")

            if drug:
                sections.append(("Druggability", "; ".join(drug)))

            # ── Chemical interactions (CTDbase) ──
            ctd_summary = []
            total_ctd = 0
            for col in cols:
                if col.startswith("ctd_") and pd.notna(row[col]) and row[col] > 0:
                    total_ctd += int(row[col])
                    short_name = CTD_PREFIXES.get(col, col.replace("ctd_", "").replace("^", " "))
                    if row[col] >= 5:
                        ctd_summary.append(f"{short_name} ({int(row[col])} chemicals)")
            if total_ctd > 0:
                top_interactions = ctd_summary[:5]
                sections.append(("Chemical-gene interactions",
                    f"{total_ctd} total CTDbase interactions" +
                    (f"; top: {', '.join(top_interactions)}" if top_interactions else "")))

            # ── Network ──
            network = []
            ppi = row.get("ppiCount", np.nan)
            if pd.notna(ppi) and ppi > 0:
                network.append(f"{int(ppi)} protein-protein interactions")

            string_l1 = row.get("string_db_L1_protein_seed_genes_overlap", np.nan)
            if pd.notna(string_l1) and string_l1 > 0:
                network.append(f"STRING L1 overlap={string_l1:.2f}")

            if network:
                sections.append(("Network", "; ".join(network)))

            # ── Protein domains (InterPro) ──
            domains = []
            for col in cols:
                if col.startswith("IPR_") and pd.notna(row[col]) and row[col] > 0:
                    domain_name = col.replace("IPR_d_", "").replace("IPR_f_", "").replace("IPR_sf_", "")
                    domains.append(domain_name)
            if domains:
                sections.append(("Protein domains", ", ".join(domains[:10]) +
                    (f" (+{len(domains)-10} more)" if len(domains) > 10 else "")))

            # ── Expression ──
            expression = []
            gtex = row.get("GTEx_spec", np.nan)
            if pd.notna(gtex):
                if gtex > 0.8:
                    expression.append("highly tissue-specific expression")
                elif gtex > 0.4:
                    expression.append("moderately tissue-specific expression")
                else:
                    expression.append("broadly expressed across tissues")

            brain = row.get("MouseGenes_Overexpressed_in_the_brain", np.nan)
            if pd.notna(brain) and brain > 0:
                expression.append("overexpressed in brain")

            if expression:
                sections.append(("Expression", "; ".join(expression)))

            # ── Assemble text profile ──
            text_parts = [f"Gene: {gene}"]
            for title, content in sections:
                text_parts.append(f"  {title}: {content}")

            text_profile = "\n".join(text_parts)

            record = {
                "gene": gene,
                "text_profile": text_profile,
                "n_sections": int(len(sections)),
                "has_pli": bool(pd.notna(pli)),
                "has_gwas": bool(pd.notna(gwas_hits) and gwas_hits > 0),
                "has_domains": bool(len(domains) > 0),
                "n_ctd": int(total_ctd),
            }
            fout.write(json.dumps(record) + "\n")

            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(todo)} done")

    # Summary
    n_total = len(existing) + len(todo)
    print(f"\n[STATIC AGENT] Done. {n_total} gene text profiles saved to {outpath}")

    # Show example
    with open(outpath) as f:
        example = json.loads(f.readline())
    print(f"\nExample ({example['gene']}):")
    print(example["text_profile"])


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Dynamic Agent - Collect external information
# ══════════════════════════════════════════════════════════════════════════════
def stage_dynamic(args):
    """Collect additional text descriptions from external sources (NCBI, UniProt)."""
    print("[DYNAMIC AGENT] Loading existing data...")

    # Load NCBI gene summaries (already fetched)
    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    g2summary = dict(zip(summaries["Gene_Symbol"].astype(str),
                         summaries["summary"].astype(str)))

    # Load static profiles
    static_profiles = {}
    static_path = os.path.join(args.outdir, "gene_text_profiles.jsonl")
    with open(static_path) as f:
        for line in f:
            obj = json.loads(line)
            static_profiles[obj["gene"]] = obj["text_profile"]

    genes = sorted(static_profiles.keys())
    print(f"  {len(genes)} genes with static profiles")
    print(f"  {len(g2summary)} genes with NCBI summaries")

    # Merge: static profile + NCBI summary + (future: UniProt, literature, etc.)
    outpath = os.path.join(args.outdir, "gene_enriched_profiles.jsonl")
    existing = set()
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                existing.add(json.loads(line)["gene"])
        print(f"  Resuming: {len(existing)} already done")

    todo = [g for g in genes if g not in existing]
    print(f"  Enriching {len(todo)} genes...")

    with open(outpath, "a") as fout:
        for idx, gene in enumerate(todo):
            profile_parts = []

            # Part 1: Static features (from gene_features.tsv)
            static_text = static_profiles.get(gene, "")
            if static_text:
                profile_parts.append(f"[Structured Database Features]\n{static_text}")

            # Part 2: NCBI gene summary (external knowledge)
            ncbi_summary = g2summary.get(gene, "")
            if ncbi_summary and ncbi_summary != "nan":
                profile_parts.append(f"[NCBI Gene Summary]\n{ncbi_summary}")

            # Part 3: Placeholder for future external API calls
            # - UniProt function description
            # - Recent PubMed abstracts
            # - Open Targets evidence
            # - DrugBank interactions
            # These can be added without changing the pipeline structure

            enriched_text = "\n\n".join(profile_parts)

            record = {
                "gene": gene,
                "enriched_text": enriched_text,
                "has_static": bool(static_text),
                "has_ncbi": bool(ncbi_summary and ncbi_summary != "nan"),
                "text_length": len(enriched_text),
            }
            fout.write(json.dumps(record) + "\n")

            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(todo)} done")

    print(f"\n[DYNAMIC AGENT] Done. Enriched profiles saved to {outpath}")

    # Stats
    n_with_ncbi = sum(1 for g in genes if g in g2summary and g2summary[g] != "nan")
    print(f"  Genes with NCBI summary: {n_with_ncbi}/{len(genes)}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: Extract structured attributes from enriched text
# ══════════════════════════════════════════════════════════════════════════════
def stage_extract(args):
    """Use LLM to extract structured attributes from enriched text profiles."""
    print("[EXTRACT] Loading enriched profiles...")
    profiles = {}
    with open(os.path.join(args.outdir, "gene_enriched_profiles.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            profiles[obj["gene"]] = obj["enriched_text"]

    genes = sorted(profiles.keys())
    print(f"  {len(genes)} genes")

    # Use Azure GPT-4o-mini for extraction
    from llm.llm_backends import AzureOpenAIBackend
    def _key(fname):
        path = os.path.expanduser(f"~/.secrets/{fname}")
        return open(path).read().strip()

    backend = AzureOpenAIBackend(
        chat_endpoint="https://duanz-mkv4bpb9-eastus2.cognitiveservices.azure.com",
        chat_deployment="gpt-4o-mini",
        chat_api_version="2024-02-01",
        chat_api_key=_key("azure_openai_eastus2_key.txt"),
        embed_endpoint="https://duanz-mkv4bpb9-westus.cognitiveservices.azure.com",
        embed_deployment="text-embedding-3-large",
        embed_api_version="2024-02-01",
        embed_api_key=_key("azure_openai_westus_key.txt"),
    )

    SYSTEM = """You are a drug discovery scientist. Given a comprehensive gene profile
(including structured database features AND functional descriptions), assess this gene's
therapeutic potential. Output a JSON with attribute scores in [0.0, 1.0].
You should include ANY attributes you find relevant. Consider both the numerical database
features AND the text descriptions to form a holistic assessment.
Return STRICT JSON only."""

    outpath = os.path.join(args.outdir, "agent_attributes.csv")

    # Resume
    existing = {}
    if os.path.exists(outpath):
        df_existing = pd.read_csv(outpath)
        existing = set(df_existing["Gene_Symbol"].astype(str))
        print(f"  Resuming: {len(existing)} already done")

    todo = [g for g in genes if g not in existing]
    print(f"  Extracting attributes for {len(todo)} genes...")

    import re
    results = []
    if os.path.exists(outpath):
        results = pd.read_csv(outpath).to_dict("records")

    for idx, gene in enumerate(todo):
        enriched = profiles[gene]
        # Truncate if too long
        if len(enriched) > 3000:
            enriched = enriched[:3000] + "\n[truncated]"

        prompt = f"""Analyze this gene's therapeutic target potential based on ALL available evidence:

{enriched}

Return a JSON with therapeutic attribute scores in [0.0, 1.0]. Include any attributes
you find relevant (druggability, essentiality, disease association, selectivity, etc.)."""

        try:
            response = backend._call_llm(system=SYSTEM, user=prompt)
            m = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if m:
                attrs = json.loads(m.group())
                row = {"Gene_Symbol": gene}
                for k, v in attrs.items():
                    try:
                        val = float(v)
                        if 0 <= val <= 1:
                            row[k] = val
                    except (ValueError, TypeError):
                        pass
                results.append(row)
        except Exception as e:
            if (idx + 1) % 100 == 0:
                print(f"  Error for {gene}: {e}")

        if (idx + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(outpath, index=False)
            print(f"  {idx+1}/{len(todo)} done, {len(results)} valid, saved")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(outpath, index=False)
    attr_cols = [c for c in df.columns if c != "Gene_Symbol"]
    print(f"\n[EXTRACT] Done. {len(df)} genes, {len(attr_cols)} unique attributes")
    print(f"  Top attributes: {attr_cols[:20]}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["static", "dynamic", "extract", "all"])
    ap.add_argument("--bio_features", default="data/gene_features.tsv")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--outdir", default="results/agent_pipeline")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.stage in ("static", "all"):
        stage_static(args)
    if args.stage in ("dynamic", "all"):
        stage_dynamic(args)
    if args.stage in ("extract", "all"):
        stage_extract(args)


if __name__ == "__main__":
    main()
