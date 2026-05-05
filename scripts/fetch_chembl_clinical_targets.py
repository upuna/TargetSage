#!/usr/bin/env python3
"""
Fetch human protein targets that have drugs in Phase 2 or 3 clinical trials
from ChEMBL mechanism endpoint (direct phase filter).
Outputs: data/chembl_clinical_targets_phase23.csv
"""

import time, requests, pandas as pd

BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get_json(url, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def fetch_mechanisms(min_phase=2, max_phase=3):
    print(f"[1/2] Fetching mechanisms with max_phase {min_phase}-{max_phase} ...")
    rows, offset, limit = [], 0, 1000
    while True:
        data = get_json(f"{BASE}/mechanism",
                        params={"max_phase__gte": min_phase,
                                "max_phase__lte": max_phase,
                                "format": "json",
                                "limit": limit,
                                "offset": offset})
        if data is None:
            break
        batch = data.get("mechanisms", [])
        for m in batch:
            tid = m.get("target_chembl_id")
            phase = m.get("max_phase") or 0
            if tid:
                rows.append({"target_chembl_id": tid, "max_phase": float(phase)})
        total = data.get("page_meta", {}).get("total_count", "?")
        print(f"  {offset + len(batch)}/{total} mechanisms ...", end="\r")
        if len(batch) < limit:
            break
        offset += limit
    print()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.groupby("target_chembl_id", as_index=False)["max_phase"].max()
    print(f"  unique targets: {len(df)}")
    return df


def resolve_gene_symbols(target_df):
    print("[2/2] Resolving target -> gene symbol ...")
    tids = target_df["target_chembl_id"].tolist()
    phase_map = dict(zip(target_df["target_chembl_id"], target_df["max_phase"]))
    rows, batch_size = [], 50
    for i in range(0, len(tids), batch_size):
        batch = tids[i:i + batch_size]
        ids_str = ",".join(batch)
        data = get_json(f"{BASE}/target",
                        params={"target_chembl_id__in": ids_str,
                                "organism": "Homo sapiens",
                                "target_type": "SINGLE PROTEIN",
                                "format": "json",
                                "limit": 200})
        if data is None:
            continue
        for t in data.get("targets", []):
            tid = t["target_chembl_id"]
            for comp in t.get("target_components", []):
                for xref in comp.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "HGNC" and xref.get("xref_name"):
                        rows.append({
                            "gene_symbol": xref["xref_name"].upper(),
                            "chembl_target_id": tid,
                            "max_phase": phase_map.get(tid, 0),
                        })
        if (i // batch_size) % 10 == 0:
            print(f"  {i}/{len(tids)} targets resolved ...", end="\r")
    print()
    return pd.DataFrame(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_phase", type=float, default=2.0)
    ap.add_argument("--max_phase", type=float, default=3.0)
    ap.add_argument("--out", default="data/chembl_clinical_targets_phase23.csv")
    args = ap.parse_args()

    mech_df = fetch_mechanisms(args.min_phase, args.max_phase)
    if mech_df.empty:
        print("[ERROR] no mechanisms found"); return

    gene_df = resolve_gene_symbols(mech_df)
    if gene_df.empty:
        print("[ERROR] no gene symbols resolved"); return

    gene_df = (gene_df.sort_values("max_phase", ascending=False)
                      .drop_duplicates("gene_symbol")
                      .reset_index(drop=True))

    gene_df.to_csv(args.out, index=False)
    print(f"[DONE] {len(gene_df)} genes -> {args.out}")
    print(gene_df.head(10).to_string())


if __name__ == "__main__":
    main()
