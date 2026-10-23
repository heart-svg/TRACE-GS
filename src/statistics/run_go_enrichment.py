#!/usr/bin/env python3
"""GO enrichment for predeclared cross-fold-stable RNA genes.

Uses g:Profiler's public API with the assayed RNA-gene universe supplied as a
custom background.  Stable genes are selected upstream without test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests


API = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
ANALYSES = {
    "Arabidopsis_FT10": ("athaliana", "rna_stability_arabidopsis/FT10"),
    "Arabidopsis_FT16": ("athaliana", "rna_stability_arabidopsis/FT16"),
    "Maize_Headingdate": ("zmays", "rna_stability_maize_headingdate"),
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, (organism, rel) in ANALYSES.items():
        d = args.root / rel
        stable_path = d / "stable_genes_primary.csv"
        universe_path = d / "gene_stability_summary.csv"
        stable = pd.read_csv(stable_path)["gene"].astype(str).drop_duplicates().tolist()
        universe = pd.read_csv(universe_path)["gene"].astype(str).drop_duplicates().tolist()
        payload = {
            "organism": organism,
            "query": stable,
            "sources": ["GO:BP", "GO:MF", "GO:CC"],
            "user_threshold": 0.05,
            "significance_threshold_method": "fdr",
            "domain_scope": "custom",
            "background": universe,
            "no_evidences": False,
        }
        response = requests.post(API, json=payload, timeout=180)
        response.raise_for_status()
        raw = response.json()
        (args.output_dir / f"{label}_gprofiler_raw.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        results = pd.DataFrame(raw.get("result", []))
        if len(results):
            keep = [c for c in ["source", "native", "name", "p_value", "term_size", "query_size",
                                  "intersection_size", "effective_domain_size", "intersection"] if c in results]
            results = results[keep].sort_values(["p_value", "source", "native"])
        results.to_csv(args.output_dir / f"{label}_GO_enrichment.csv", index=False)
        report[label] = {
            "organism": organism,
            "stable_gene_count": len(stable),
            "custom_background_gene_count": len(universe),
            "significant_terms_fdr_lt_0.05": len(results),
            "stable_source_sha256": sha256(stable_path),
            "background_source_sha256": sha256(universe_path),
            "api": API,
            "sources": ["GO:BP", "GO:MF", "GO:CC"],
            "multiple_testing": "g:Profiler FDR, threshold 0.05",
        }
    (args.output_dir / "GO_ANALYSIS_COMPLETE.json").write_text(
        json.dumps({"status": "PASS", "analyses": report}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

