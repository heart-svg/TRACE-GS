#!/usr/bin/env python3
"""GO enrichment for maize GRMZM2G genes using version-matched MaizeGDB data.

The formal maize expression matrix uses B73 RefGen_v3/5b+ GRMZM2G identifiers.
This script therefore avoids lossy conversion to current B73v5 identifiers.  It
uses the official MaizeGDB B73_RefGen_v3-GMs-GOTerms table, restricts both query
and population to assayed RNA genes, applies one-sided hypergeometric tests,
and controls the Benjamini-Hochberg FDR across all tested GO terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import hypergeom


GO_RE = re.compile(r"(GO:\d{7}):([^\"}]+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    n = len(pvalues)
    order = pvalues.sort_values().index
    ranked = pvalues.loc[order].to_numpy()
    adjusted = ranked * n / pd.Series(range(1, n + 1)).to_numpy()
    adjusted = pd.Series(adjusted[::-1]).cummin().to_numpy()[::-1]
    out = pd.Series(index=order, data=adjusted.clip(max=1.0))
    return out.reindex(pvalues.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stable", type=Path, required=True)
    ap.add_argument("--universe", type=Path, required=True)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-term-size", type=int, default=5)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stable = set(pd.read_csv(args.stable)["gene"].astype(str))
    universe = set(pd.read_csv(args.universe)["gene"].astype(str))
    ann = pd.read_csv(args.annotation, usecols=["gene_model", "obo_terms"])
    ann["gene_model"] = ann["gene_model"].astype(str)

    term_to_genes: dict[str, set[str]] = {}
    term_names: dict[str, str] = {}
    for gene, raw in ann.itertuples(index=False):
        if gene not in universe or not isinstance(raw, str):
            continue
        for go_id, name in GO_RE.findall(raw):
            term_to_genes.setdefault(go_id, set()).add(gene)
            term_names.setdefault(go_id, name.strip())

    annotated_universe = set().union(*term_to_genes.values()) if term_to_genes else set()
    query = stable & annotated_universe
    M = len(annotated_universe)
    N = len(query)
    rows = []
    for go_id, genes in term_to_genes.items():
        n = len(genes)
        overlap = sorted(query & genes)
        k = len(overlap)
        # Every GO term meeting the population-size rule enters the BH family,
        # including terms with zero/one query hit. Filtering on overlap before
        # correction would make the reported FDR anti-conservative.
        if n < args.min_term_size:
            continue
        p = float(hypergeom.sf(k - 1, M, n, N))
        rows.append({
            "source": "GO",
            "native": go_id,
            "name": term_names.get(go_id, ""),
            "p_value": p,
            "term_size_in_assayed_annotated_background": n,
            "query_size_annotated": N,
            "intersection_size": k,
            "effective_domain_size": M,
            "intersection": ";".join(overlap),
        })

    results = pd.DataFrame(rows)
    if len(results):
        results["p_adjust_bh"] = bh_adjust(results["p_value"])
        results = results.sort_values(["p_adjust_bh", "p_value", "native"])
        results["significant_fdr_0.05"] = results["p_adjust_bh"] < 0.05
    results.to_csv(args.output_dir / "Maize_Headingdate_GO_enrichment_refgenv3.csv", index=False)

    summary = {
        "status": "PASS",
        "species": "Zea mays",
        "trait": "Headingdate",
        "gene_id_namespace": "B73 RefGen_v3 5b+ GRMZM2G",
        "annotation_source": "MaizeGDB B73_RefGen_v3-GMs-GOTerms.csv.gz",
        "annotation_url": "https://download.maizegdb.org/B73_RefGen_v3/B73_RefGen_v3-GMs-GOTerms.csv.gz",
        "test": "one-sided hypergeometric overrepresentation",
        "multiple_testing": "Benjamini-Hochberg across all tested GO terms",
        "stable_gene_count_total": len(stable),
        "assayed_gene_count_total": len(universe),
        "annotated_assayed_background_count": M,
        "annotated_stable_gene_count": N,
        "tested_term_count": len(results),
        "significant_term_count_fdr_lt_0.05": int(results.get("significant_fdr_0.05", pd.Series(dtype=bool)).sum()),
        "min_term_size": args.min_term_size,
        "minimum_query_overlap_for_testing": 0,
        "source_sha256": {
            "stable": sha256(args.stable),
            "universe": sha256(args.universe),
            "annotation": sha256(args.annotation),
        },
        "interpretation_guardrail": "No claim of enrichment is made for unannotated GRMZM2G genes; no mapping to B73v5 was used.",
    }
    (args.output_dir / "Maize_Headingdate_GO_COMPLETE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

