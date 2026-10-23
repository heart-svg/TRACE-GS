"""Build an evidence-bounded FT10 outer-0 efficiency table.

This script deliberately separates measured training time from prediction-only
legacy measurements.  DEM/DEM-Fix training cells remain missing until a true
from-scratch run under the same timing boundary is available.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "raw" / "FT10_outer0_native_v2"
OUT = ROOT / "submission_efficiency_table_draft.csv"


def native_records():
    models = ["LASSO", "SVR", "XGBoost", "DeepGS", "LCNN", "Cropformer"]
    for model in models:
        records = []
        for p in sorted(NATIVE.glob(f"{model}_repeat*.json")):
            records.append(json.loads(p.read_text(encoding="utf-8")))
        if not records:
            continue
        warm = records[1:] if len(records) > 1 else records
        mean_train = sum(float(x["training_seconds"]) for x in warm) / len(warm)
        mean_latency = 1000 * sum(float(x["single_sample_warm_latency_seconds_median"]) for x in warm) / len(warm)
        row0 = records[0]
        if model == "SVR":
            complexity = f"{int(row0.get('parameter_count_or_fitted_coefficient_count', 0))} stored SV coefficients/elements"
        elif model == "XGBoost":
            complexity = "tree ensemble; trainable-parameter count not defined"
        else:
            pcount = row0.get("parameter_count_or_fitted_coefficient_count")
            complexity = str(int(pcount)) if pcount is not None else "NA"
        yield {
            "model": model,
            "input": f"{row0.get('input', '')}; {row0.get('n_features', 'NA')} features",
            "training_seconds": round(mean_train, 3),
            "single_sample_latency_ms": round(mean_latency, 3),
            "complexity_measure": complexity,
            "timing_repeats": len(warm),
            "status": "MEASURED_COMPARABLE",
            "note": "cold run excluded; locked FT10 outer0; HPO excluded",
        }


rows = list(native_records())
rows.extend(
    [
        {
            "model": "BayesA",
            "input": "819681 QC SNPs",
            "training_seconds": 1767.821,
            "single_sample_latency_ms": "NA",
            "complexity_measure": "819681 marker effects",
            "timing_repeats": 1,
            "status": "MEASURED_SINGLE_RUN",
            "note": "official BGLR; 12000 iterations; HPO excluded",
        },
        {
            "model": "BayesB",
            "input": "819681 QC SNPs",
            "training_seconds": 1988.295,
            "single_sample_latency_ms": "NA",
            "complexity_measure": "819681 marker effects",
            "timing_repeats": 1,
            "status": "MEASURED_SINGLE_RUN",
            "note": "official BGLR; 12000 iterations; HPO excluded",
        },
        {
            "model": "BayesC",
            "input": "819681 QC SNPs",
            "training_seconds": 1662.741,
            "single_sample_latency_ms": "NA",
            "complexity_measure": "819681 marker effects",
            "timing_repeats": 1,
            "status": "MEASURED_SINGLE_RUN",
            "note": "official BGLR; 12000 iterations; HPO excluded",
        },
        {
            "model": "rrBLUP",
            "input": "precomputed additive relationship matrix from 819681 QC SNPs",
            "training_seconds": "NA",
            "single_sample_latency_ms": "NA",
            "complexity_measure": "645 x 645 relationship matrix",
            "timing_repeats": 0,
            "status": "REMEASURE_REQUIRED",
            "note": "existing 0.206 s covers five fits but excludes relationship-matrix construction",
        },
        {
            "model": "TRACE-GS DNA backbone",
            "input": "fold-specific 26867-gene S2G representation",
            "training_seconds": 90.482,
            "single_sample_latency_ms": "NA",
            "complexity_measure": "96885 trainable parameters",
            "timing_repeats": 3,
            "status": "MEASURED_COMPARABLE",
            "note": "cold run excluded; exact formal trainer; HPO excluded",
        },
    ]
)
for model in ["DEM DNA-only", "DEM DNA+RNA", "DEM-Fix DNA-only", "DEM-Fix DNA+RNA"]:
    rows.append(
        {
            "model": model,
            "input": "formal FT10 outer0 litdata",
            "training_seconds": "NA",
            "single_sample_latency_ms": "NA",
            "complexity_measure": "NA",
            "timing_repeats": 0,
            "status": "TRAINING_REMEASURE_REQUIRED",
            "note": "legacy evidence is prediction-only and is excluded",
        }
    )

fields = list(rows[0])
with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(OUT)

