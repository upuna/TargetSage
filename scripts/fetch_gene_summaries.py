#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch NCBI gene summaries for all genes in gene_labels.tsv.

Usage:
    python scripts/fetch_gene_summaries.py \
        --labels data/gene_labels.tsv \
        --out    data/gene_summaries.tsv \
        --email  your@email.com

    # Resume if interrupted
    python scripts/fetch_gene_summaries.py --resume

Requirements:
    pip install biopython pandas
"""

import os
import sys
import time
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def entrez_search_gene_id(symbol: str, email: str):
    from Bio import Entrez
    Entrez.email = email
    try:
        handle = Entrez.esearch(
            db="gene",
            term=f"{symbol}[Gene Name] AND 9606[Taxonomy ID] AND alive[prop]",
            retmax=1,
        )
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        return ids[0] if ids else None
    except Exception as e:
        log.warning(f"esearch failed for {symbol}: {e}")
        return None


def entrez_fetch_summary(gene_id: str, email: str) -> str:
    from Bio import Entrez
    Entrez.email = email
    try:
        handle  = Entrez.esummary(db="gene", id=gene_id, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        doc      = records["DocumentSummarySet"]["DocumentSummary"][0]
        summary  = str(doc.get("Summary", "")).strip()
        fullname = str(doc.get("Description", "")).strip()
        return summary if summary else fullname
    except Exception as e:
        log.warning(f"esummary failed for gene_id={gene_id}: {e}")
        return ""


def fetch_one(symbol: str, email: str) -> str:
    gene_id = entrez_search_gene_id(symbol, email)
    if not gene_id:
        return ""
    return entrez_fetch_summary(gene_id, email)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--labels",    default="data/gene_labels.tsv")
    ap.add_argument("--out",       default="data/gene_summaries.tsv")
    ap.add_argument("--email",     default="your@email.com",
                    help="Required by NCBI Entrez policy")
    ap.add_argument("--delay",     type=float, default=0.34,
                    help="Seconds between API calls (NCBI limit: ~3/sec)")
    ap.add_argument("--resume",    action="store_true",
                    help="Skip genes already in output file")
    ap.add_argument("--max_genes", type=int, default=0,
                    help="Debug: process only first N genes (0=all)")
    args = ap.parse_args()

    try:
        from Bio import Entrez  # noqa
    except ImportError:
        sys.exit("Run: pip install biopython")

    # Load gene list
    df = pd.read_csv(args.labels, sep="\t")
    if "Gene_Symbol" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Gene_Symbol"})
    genes = df["Gene_Symbol"].dropna().unique().tolist()
    if args.max_genes > 0:
        genes = genes[:args.max_genes]
    log.info(f"{len(genes)} genes to process")

    # Load existing if resuming
    done = {}
    if args.resume and os.path.isfile(args.out):
        existing = pd.read_csv(args.out, sep="\t")
        done = dict(zip(existing["Gene_Symbol"], existing["summary"]))
        log.info(f"[resume] {len(done)} genes already done")

    results = dict(done)
    todo    = [g for g in genes if g not in done]
    log.info(f"Fetching {len(todo)} genes ...")

    for i, symbol in enumerate(todo):
        results[symbol] = fetch_one(symbol, args.email)

        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            pd.DataFrame([{"Gene_Symbol": g, "summary": s}
                          for g, s in results.items()]).to_csv(
                args.out, sep="\t", index=False)
            log.info(f"  [{i+1}/{len(todo)}] saved → {args.out}")

        time.sleep(args.delay)

    out_df  = pd.DataFrame([{"Gene_Symbol": g, "summary": results.get(g, "")}
                             for g in genes])
    n_empty = int((out_df["summary"] == "").sum())
    out_df.to_csv(args.out, sep="\t", index=False)
    log.info(f"Done. {len(out_df)} genes, {n_empty} with no summary → {args.out}")

    print("\nSample output:")
    for _, row in out_df[out_df["summary"] != ""].head(3).iterrows():
        print(f"  {row['Gene_Symbol']:<12} {row['summary'][:100]}...")


if __name__ == "__main__":
    main()