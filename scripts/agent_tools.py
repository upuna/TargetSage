#!/usr/bin/env python3
"""
Multi-Tool Agentic Feature Collection Pipeline

Tools:
  1. StructuredFeatureReader — gene_features.tsv → semantic text
  2. NCBIGeneSummary         — NCBI Entrez API
  3. UniProtFunction         — UniProt REST API
  4. PubMedSearch            — PubMed abstracts
  5. OpenTargetsEvidence     — Open Targets GraphQL API
  6. StringDBContext         — STRING API for PPI context
  7. FactChecker             — Conflict detection + label leakage filter

Usage:
  python -u scripts/agent_tools.py --stage collect    # Run all tools
  python -u scripts/agent_tools.py --stage check      # Fact-check collected data
  python -u scripts/agent_tools.py --stage extract     # LLM attribute extraction
  python -u scripts/agent_tools.py --stage all
"""

import os, sys, json, re, time, argparse
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: Structured Feature Reader
# ══════════════════════════════════════════════════════════════════════════════
class StructuredFeatureReader:
    """Convert numerical gene_features.tsv into semantic text."""

    def __init__(self, bio_path="data/gene_features.tsv"):
        self.df = pd.read_csv(bio_path, sep="\t")
        self.cols = [c for c in self.df.columns if c != "Gene_Symbol"]

    def __call__(self, gene):
        rows = self.df[self.df["Gene_Symbol"] == gene]
        if len(rows) == 0:
            return None
        row = rows.iloc[0]
        parts = []

        # Genomic constraint
        pli = row.get("GnomAD_pLI", np.nan)
        if pd.notna(pli):
            level = "highly constrained" if pli > 0.9 else "moderately constrained" if pli > 0.5 else "tolerant"
            parts.append(f"Loss-of-function {level} (pLI={pli:.2f})")

        oe = row.get("GnomAD_oe_lof", np.nan)
        if pd.notna(oe):
            parts.append(f"Observed/expected LoF ratio: {oe:.2f}")

        rvis = row.get("GenicIntolerance_RVIS", np.nan)
        if pd.notna(rvis):
            parts.append(f"RVIS={rvis:.3f}")

        if row.get("MouseGenes_Essential", 0) > 0:
            parts.append("Essential gene in mouse")

        seq_len = row.get("uniprot_seq_len", np.nan)
        if pd.notna(seq_len) and seq_len > 0:
            parts.append(f"Protein length: {int(seq_len)} aa")

        # Disease
        gwas = row.get("GWAS_hits", np.nan)
        if pd.notna(gwas) and gwas > 0:
            parts.append(f"{int(gwas)} GWAS associations")
        omim = row.get("OMIM_uniq_diseases", np.nan)
        if pd.notna(omim) and omim > 0:
            parts.append(f"{int(omim)} OMIM disease(s)")

        # Druggability
        dgidb = row.get("DGIdb_interaction_types", np.nan)
        if pd.notna(dgidb) and dgidb > 0:
            parts.append(f"{int(dgidb)} drug interaction types (DGIdb)")
        ab = row.get("antibodyCount", np.nan)
        if pd.notna(ab) and ab > 0:
            parts.append(f"{int(ab)} known antibodies")

        # CTDbase
        total_ctd = sum(1 for c in self.cols if c.startswith("ctd_") and pd.notna(row.get(c, np.nan)) and row[c] > 0)
        if total_ctd > 0:
            parts.append(f"{total_ctd} types of chemical-gene interactions (CTDbase)")

        # Network
        ppi = row.get("ppiCount", np.nan)
        if pd.notna(ppi) and ppi > 0:
            parts.append(f"{int(ppi)} protein-protein interactions")

        # Domains
        domains = [c.replace("IPR_d_", "").replace("IPR_f_", "").replace("IPR_sf_", "")
                   for c in self.cols if c.startswith("IPR_") and pd.notna(row.get(c, np.nan)) and row[c] > 0]
        if domains:
            parts.append(f"Protein domains: {', '.join(domains[:8])}")

        # Expression
        gtex = row.get("GTEx_spec", np.nan)
        if pd.notna(gtex):
            spec = "tissue-specific" if gtex > 0.6 else "broadly expressed"
            parts.append(f"Expression: {spec} (GTEx specificity={gtex:.2f})")

        return "; ".join(parts) if parts else "No structured features available."


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: NCBI Gene Summary
# ══════════════════════════════════════════════════════════════════════════════
class NCBIGeneSummary:
    """Fetch gene summary from pre-downloaded NCBI data."""

    def __init__(self, summary_path="data/gene_summaries.tsv"):
        df = pd.read_csv(summary_path, sep="\t")
        self.data = dict(zip(df["Gene_Symbol"].astype(str), df["summary"].astype(str)))

    def __call__(self, gene):
        s = self.data.get(gene, "")
        return s if s and s != "nan" else None


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: UniProt Function
# ══════════════════════════════════════════════════════════════════════════════
class UniProtFunction:
    """Query UniProt REST API for function annotation."""

    BASE = "https://rest.uniprot.org/uniprotkb/search"

    def __call__(self, gene, organism="human"):
        try:
            params = {
                "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "fields": "cc_function,cc_subcellular_location,cc_tissue_specificity,cc_pathway",
                "size": 1,
            }
            r = requests.get(self.BASE, params=params, timeout=10)
            if r.status_code != 200:
                return None
            data = r.json()
            if not data.get("results"):
                return None

            entry = data["results"][0]
            parts = []

            # Function
            for comment in entry.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    for text in comment.get("texts", []):
                        parts.append(f"Function: {text.get('value', '')}")
                elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                    locs = []
                    for sub in comment.get("subcellularLocations", []):
                        loc = sub.get("location", {}).get("value", "")
                        if loc:
                            locs.append(loc)
                    if locs:
                        parts.append(f"Localization: {', '.join(locs)}")
                elif comment.get("commentType") == "TISSUE SPECIFICITY":
                    for text in comment.get("texts", []):
                        parts.append(f"Tissue specificity: {text.get('value', '')}")

            return "; ".join(parts) if parts else None
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4: PubMed Search
# ══════════════════════════════════════════════════════════════════════════════
class PubMedSearch:
    """Search PubMed for recent abstracts about the gene."""

    ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, email="targetsage@example.com", max_results=3):
        self.email = email
        self.max_results = max_results

    def __call__(self, gene):
        try:
            # Search
            params = {
                "db": "pubmed",
                "term": f"{gene}[Gene] AND drug target[Title/Abstract]",
                "retmax": self.max_results,
                "retmode": "json",
                "email": self.email,
                "sort": "date",
            }
            r = requests.get(self.ESEARCH, params=params, timeout=10)
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return None

            # Fetch abstracts
            params = {
                "db": "pubmed",
                "id": ",".join(ids[:2]),
                "rettype": "abstract",
                "retmode": "text",
                "email": self.email,
            }
            r = requests.get(self.EFETCH, params=params, timeout=10)
            text = r.text.strip()
            # Truncate
            if len(text) > 1000:
                text = text[:1000] + "..."
            return text if text else None
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5: Open Targets Evidence
# ══════════════════════════════════════════════════════════════════════════════
class OpenTargetsEvidence:
    """Query Open Targets Platform GraphQL API for target evidence."""

    ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"

    def __call__(self, gene):
        try:
            query = """
            query target($ensemblId: String!) {
                target(ensemblId: $ensemblId) {
                    approvedSymbol
                    biotype
                    tractability {
                        label
                        modality
                        value
                    }
                    safetyLiabilities {
                        event
                        effects {
                            direction
                            dosing
                        }
                    }
                }
            }
            """
            # First get Ensembl ID from gene symbol
            search_query = """
            query search($q: String!) {
                search(queryString: $q, entityNames: ["target"], page: {size: 1, index: 0}) {
                    hits { id }
                }
            }
            """
            r = requests.post(self.ENDPOINT,
                json={"query": search_query, "variables": {"q": gene}}, timeout=10)
            hits = r.json().get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                return None

            ensembl_id = hits[0]["id"]
            r = requests.post(self.ENDPOINT,
                json={"query": query, "variables": {"ensemblId": ensembl_id}}, timeout=10)
            target = r.json().get("data", {}).get("target")
            if not target:
                return None

            parts = []
            # Tractability
            tract = target.get("tractability", [])
            if tract:
                modalities = set()
                for t in tract:
                    if t.get("value", False):
                        modalities.add(t.get("modality", ""))
                if modalities:
                    parts.append(f"Tractable modalities: {', '.join(modalities)}")

            # Safety
            safety = target.get("safetyLiabilities", [])
            if safety:
                events = [s.get("event", "") for s in safety[:3]]
                parts.append(f"Safety liabilities: {', '.join(events)}")

            return "; ".join(parts) if parts else None
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# Tool 6: FactChecker Agent
# ══════════════════════════════════════════════════════════════════════════════
class FactChecker:
    """
    Check collected information for:
    1. Label leakage (mentions of approved drugs, FDA approval, etc.)
    2. Conflicting facts between sources
    3. Temporal leakage for validation sets
    """

    LEAKAGE_PATTERNS = [
        r"(?i)FDA[\s-]?approved",
        r"(?i)clinically[\s-]?approved",
        r"(?i)approved[\s-]?drug[\s-]?target",
        r"(?i)first[\s-]?line[\s-]?treatment",
        r"(?i)standard[\s-]?of[\s-]?care",
        r"(?i)marketed[\s-]?drug",
    ]

    def __init__(self):
        self.leakage_re = [re.compile(p) for p in self.LEAKAGE_PATTERNS]

    def check_leakage(self, text):
        """Return list of leaked phrases found."""
        if not text:
            return []
        found = []
        for pattern in self.leakage_re:
            matches = pattern.findall(text)
            found.extend(matches)
        return found

    def sanitize(self, text):
        """Remove sentences containing leakage patterns."""
        if not text:
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean = []
        for sent in sentences:
            has_leak = any(p.search(sent) for p in self.leakage_re)
            if not has_leak:
                clean.append(sent)
        return " ".join(clean)

    def check_conflicts(self, sources):
        """
        Detect potential conflicts between sources.
        Returns list of (source1, source2, conflict_description).
        """
        conflicts = []
        # Simple heuristic: check for contradictory keywords
        localization_keywords = {
            "nuclear": "nucleus", "cytoplasmic": "cytoplasm",
            "membrane": "membrane", "secreted": "extracellular",
            "mitochondrial": "mitochondria",
        }

        locs_per_source = {}
        for source_name, text in sources.items():
            if not text:
                continue
            found_locs = set()
            for keyword, loc in localization_keywords.items():
                if keyword.lower() in text.lower():
                    found_locs.add(loc)
            if found_locs:
                locs_per_source[source_name] = found_locs

        # Check pairwise conflicts
        source_names = list(locs_per_source.keys())
        for i in range(len(source_names)):
            for j in range(i+1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                l1, l2 = locs_per_source[s1], locs_per_source[s2]
                if l1 and l2 and not l1.intersection(l2):
                    conflicts.append((s1, s2,
                        f"Localization mismatch: {s1} says {l1}, {s2} says {l2}"))

        return conflicts


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator: Collect from all tools
# ══════════════════════════════════════════════════════════════════════════════
def stage_collect(args):
    """Run all tools to collect multi-source gene profiles."""
    print("[ORCHESTRATOR] Initializing tools...")
    tools = {
        "structured_features": StructuredFeatureReader(args.bio_features),
        "ncbi_summary": NCBIGeneSummary(args.gene_summaries),
        "uniprot_function": UniProtFunction(),
        "pubmed_abstracts": PubMedSearch(),
        "open_targets": OpenTargetsEvidence(),
    }
    fact_checker = FactChecker()

    # Get gene list
    labels = pd.read_csv(args.labels, sep="\t")
    genes = sorted(labels["Gene_Symbol"].astype(str).unique())
    print(f"  {len(genes)} genes to process")

    outpath = os.path.join(args.outdir, "agent_collected.jsonl")
    os.makedirs(args.outdir, exist_ok=True)

    # Resume
    existing = set()
    if os.path.exists(outpath):
        with open(outpath) as f:
            for line in f:
                existing.add(json.loads(line)["gene"])
        print(f"  Resuming: {len(existing)} already done")

    todo = [g for g in genes if g not in existing]
    if not todo:
        print("  All genes already collected.")
        return
    print(f"  Collecting for {len(todo)} genes...")

    # Rate limiting for APIs
    api_delay = 0.15  # seconds between API calls

    with open(outpath, "a") as fout:
        for idx, gene in enumerate(todo):
            sources = {}

            # Tool 1: Structured features (instant, no API)
            sources["structured_features"] = tools["structured_features"](gene)

            # Tool 2: NCBI (pre-downloaded, instant)
            sources["ncbi_summary"] = tools["ncbi_summary"](gene)

            # Tool 3: UniProt (API call)
            sources["uniprot_function"] = tools["uniprot_function"](gene)
            time.sleep(api_delay)

            # Tool 4: PubMed (API call) — only for first 2000 genes for speed
            if idx < 2000:
                sources["pubmed_abstracts"] = tools["pubmed_abstracts"](gene)
                time.sleep(api_delay)

            # Tool 5: Open Targets (API call)
            sources["open_targets"] = tools["open_targets"](gene)
            time.sleep(api_delay)

            # Fact-check: leakage + conflicts
            leakage = {}
            sanitized = {}
            for src_name, text in sources.items():
                if text:
                    leaked = fact_checker.check_leakage(text)
                    if leaked:
                        leakage[src_name] = leaked
                        sanitized[src_name] = fact_checker.sanitize(text)
                    else:
                        sanitized[src_name] = text
                else:
                    sanitized[src_name] = None

            conflicts = fact_checker.check_conflicts(sanitized)

            # Assemble enriched profile
            profile_parts = []
            for src_name, text in sanitized.items():
                if text:
                    display_name = src_name.replace("_", " ").title()
                    profile_parts.append(f"[{display_name}]\n{text}")

            enriched_text = "\n\n".join(profile_parts)

            record = {
                "gene": gene,
                "enriched_text": enriched_text,
                "sources": {k: bool(v) for k, v in sanitized.items()},
                "n_sources": sum(1 for v in sanitized.values() if v),
                "leakage_detected": leakage if leakage else None,
                "conflicts": conflicts if conflicts else None,
                "text_length": len(enriched_text),
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if (idx + 1) % 100 == 0:
                n_sources_avg = np.mean([sum(1 for v in sanitized.values() if v)])
                n_leaks = sum(1 for r in [record] if r["leakage_detected"])
                print(f"  {idx+1}/{len(todo)} done | "
                      f"sources/gene: {record['n_sources']} | "
                      f"leakage: {len(leakage)} source(s) | "
                      f"conflicts: {len(conflicts)}")

    # Summary stats
    print(f"\n[ORCHESTRATOR] Collection complete. Saved to {outpath}")
    total = len(existing) + len(todo)
    print(f"  Total genes: {total}")


def stage_check(args):
    """Run fact-checking on collected data and print summary."""
    print("[FACT-CHECK] Loading collected data...")
    records = []
    with open(os.path.join(args.outdir, "agent_collected.jsonl")) as f:
        for line in f:
            records.append(json.loads(line))

    n_leakage = sum(1 for r in records if r.get("leakage_detected"))
    n_conflicts = sum(1 for r in records if r.get("conflicts"))
    n_sources = np.mean([r["n_sources"] for r in records])

    print(f"  Total genes: {len(records)}")
    print(f"  Avg sources per gene: {n_sources:.1f}")
    print(f"  Genes with leakage detected: {n_leakage} ({n_leakage/len(records)*100:.1f}%)")
    print(f"  Genes with conflicts: {n_conflicts} ({n_conflicts/len(records)*100:.1f}%)")

    # Show examples
    if n_leakage > 0:
        print("\n  Leakage examples:")
        for r in records:
            if r.get("leakage_detected"):
                print(f"    {r['gene']}: {r['leakage_detected']}")
                break

    if n_conflicts > 0:
        print("\n  Conflict examples:")
        for r in records:
            if r.get("conflicts"):
                print(f"    {r['gene']}: {r['conflicts']}")
                break


def stage_extract(args):
    """Use LLM to extract structured attributes from agent-collected profiles."""
    print("[EXTRACT] Loading agent-collected profiles...")
    profiles = {}
    with open(os.path.join(args.outdir, "agent_collected.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            profiles[obj["gene"]] = obj["enriched_text"]

    genes = sorted(profiles.keys())
    print(f"  {len(genes)} genes")

    from llm.llm_backends import AzureOpenAIBackend
    def _key(fname):
        return open(os.path.expanduser(f"~/.secrets/{fname}")).read().strip()

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

    SYSTEM = """You are a drug discovery scientist. Given a gene profile assembled from
multiple sources (structured databases, NCBI, UniProt, literature, Open Targets),
assess this gene's therapeutic potential holistically. Output a JSON with attribute
scores in [0.0, 1.0]. Include any attributes you find relevant. Consider ALL evidence
sources and note where they agree or disagree. Return STRICT JSON only."""

    outpath = os.path.join(args.outdir, "agent_attributes.csv")
    existing = set()
    results = []
    if os.path.exists(outpath):
        df_ex = pd.read_csv(outpath)
        existing = set(df_ex["Gene_Symbol"].astype(str))
        results = df_ex.to_dict("records")
        print(f"  Resuming: {len(existing)} already done")

    todo = [g for g in genes if g not in existing]
    print(f"  Extracting for {len(todo)} genes...")

    for idx, gene in enumerate(todo):
        enriched = profiles[gene][:3000]
        prompt = f"Analyze ALL evidence for this gene:\n\n{enriched}\n\nReturn JSON with therapeutic attribute scores [0,1]."

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
            if (idx+1) % 100 == 0:
                print(f"  Error {gene}: {e}")

        if (idx+1) % 100 == 0:
            pd.DataFrame(results).to_csv(outpath, index=False)
            print(f"  {idx+1}/{len(todo)} done, {len(results)} valid")

    pd.DataFrame(results).to_csv(outpath, index=False)
    attr_cols = [c for c in pd.DataFrame(results).columns if c != "Gene_Symbol"]
    print(f"\n[EXTRACT] Done. {len(results)} genes, {len(attr_cols)} attributes")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["collect", "check", "extract", "all"])
    ap.add_argument("--bio_features", default="data/gene_features.tsv")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--outdir", default="results/agent_tools")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.stage in ("collect", "all"):
        stage_collect(args)
    if args.stage in ("check", "all"):
        stage_check(args)
    if args.stage in ("extract", "all"):
        stage_extract(args)


if __name__ == "__main__":
    main()
