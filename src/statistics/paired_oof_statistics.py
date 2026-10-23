#!/usr/bin/env python3
"""Paired OOF inference for TRACE-GS DNA-only versus DNA+RNA.

The independent unit is the accession/line represented by one outer-test OOF row.
No fold is treated as an independent biological replicate.  The script reports
mean-fold descriptive metrics separately from pooled-OOF paired inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAITS = {
    "Arabidopsis_FT10": "arabidopsis_FT10",
    "Arabidopsis_FT16": "arabidopsis_FT16",
    "Arabidopsis_Diameter": "arabidopsis_Diameter",
    "Maize_100grainweight": "maize_100grainweight",
    "Maize_Headingdate": "maize_Headingdate",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trait(root: Path, directory: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    d = root / directory
    rows = []
    manifest = []
    for fold in range(5):
        p = d / f"outer_{fold}_predictions.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        x = pd.read_csv(p)
        required = {"ID", "true_raw", "dna_pred_raw", "fusion_pred_raw"}
        missing = required.difference(x.columns)
        if missing:
            raise ValueError(f"{p}: missing columns {sorted(missing)}")
        x = x.copy()
        x["outer"] = fold
        rows.append(x)
        manifest.append({"file": str(p), "sha256": sha256(p), "n_rows": int(len(x))})
    oof = pd.concat(rows, ignore_index=True)
    if oof["ID"].astype(str).duplicated().any():
        dup = oof.loc[oof["ID"].astype(str).duplicated(), "ID"].astype(str).tolist()[:10]
        raise ValueError(f"duplicate OOF IDs: {dup}")
    vals = oof[["true_raw", "dna_pred_raw", "fusion_pred_raw"]].to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError(f"{directory}: non-finite y/prediction values")

    fold_rows = []
    for fold, x in oof.groupby("outer", sort=True):
        y = x["true_raw"].to_numpy(float)
        dna = x["dna_pred_raw"].to_numpy(float)
        fusion = x["fusion_pred_raw"].to_numpy(float)
        fold_rows.append(
            {
                "outer": int(fold),
                "n": int(len(x)),
                "dna_pcc": pearson(y, dna),
                "fusion_pcc": pearson(y, fusion),
                "delta_pcc": pearson(y, fusion) - pearson(y, dna),
            }
        )
    return oof, pd.DataFrame(fold_rows), manifest


def load_fish(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    p = root / "fish_stickleback_final_audit" / "fish_model_oof_predictions.csv"
    x = pd.read_csv(p)
    dna = x.loc[x["model"] == "TRACE DNA-only", ["sample_id", "outer", "observed", "prediction"]].copy()
    fusion = x.loc[x["model"] == "TRACE DNA+RNA", ["sample_id", "outer", "observed", "prediction"]].copy()
    dna = dna.rename(columns={"sample_id": "ID", "observed": "true_raw", "prediction": "dna_pred_raw"})
    fusion = fusion.rename(columns={"sample_id": "ID", "observed": "true_check", "prediction": "fusion_pred_raw"})
    oof = dna.merge(fusion, on=["ID", "outer"], how="inner", validate="one_to_one")
    if len(oof) != 351 or not np.allclose(oof["true_raw"], oof["true_check"]):
        raise ValueError("fish TRACE paired OOF reconstruction failed")
    oof = oof.drop(columns="true_check")
    fold_rows = []
    for fold, z in oof.groupby("outer", sort=True):
        y = z["true_raw"].to_numpy(float)
        a = z["dna_pred_raw"].to_numpy(float)
        b = z["fusion_pred_raw"].to_numpy(float)
        fold_rows.append({
            "outer": int(fold), "n": int(len(z)),
            "dna_pcc": pearson(y, a), "fusion_pcc": pearson(y, b),
            "delta_pcc": pearson(y, b) - pearson(y, a),
        })
    return oof, pd.DataFrame(fold_rows), [{"file": str(p), "sha256": sha256(p), "n_rows": int(len(x))}]


def load_drosophila(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    p = root / "drosophila" / "trace_pooled_oof_predictions.csv"
    oof = pd.read_csv(p, dtype={"ID": str})
    required = {"ID", "outer", "true_raw", "dna_pred_raw", "fusion_pred_raw"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"{p}: missing columns {sorted(missing)}")
    if len(oof) != 192 or oof["ID"].duplicated().any() or set(oof["outer"]) != set(range(5)):
        raise ValueError("Drosophila paired OOF audit failed")
    fold_rows = []
    for fold, z in oof.groupby("outer", sort=True):
        y = z["true_raw"].to_numpy(float)
        a = z["dna_pred_raw"].to_numpy(float)
        b = z["fusion_pred_raw"].to_numpy(float)
        fold_rows.append({
            "outer": int(fold), "n": int(len(z)),
            "dna_pcc": pearson(y, a), "fusion_pcc": pearson(y, b),
            "delta_pcc": pearson(y, b) - pearson(y, a),
        })
    return oof, pd.DataFrame(fold_rows), [{"file": str(p), "sha256": sha256(p), "n_rows": int(len(oof))}]


def paired_inference(oof: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> dict:
    y = oof["true_raw"].to_numpy(float)
    a = oof["dna_pred_raw"].to_numpy(float)
    b = oof["fusion_pred_raw"].to_numpy(float)
    n = len(y)
    obs_a = pearson(y, a)
    obs_b = pearson(y, b)
    obs = obs_b - obs_a

    rng_boot = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng_boot.integers(0, n, n)
        boot[i] = pearson(y[idx], b[idx]) - pearson(y[idx], a[idx])
    boot = boot[np.isfinite(boot)]
    if len(boot) < 0.99 * n_boot:
        raise RuntimeError("too many degenerate bootstrap replicates")

    rng_perm = np.random.default_rng(seed + 1)
    perm = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        swap = rng_perm.random(n) < 0.5
        aa = np.where(swap, b, a)
        bb = np.where(swap, a, b)
        perm[i] = pearson(y, bb) - pearson(y, aa)
    p_two = (1.0 + float(np.sum(np.abs(perm) >= abs(obs)))) / (n_perm + 1.0)
    p_gain = (1.0 + float(np.sum(perm >= obs))) / (n_perm + 1.0)

    return {
        "n_oof": int(n),
        "pooled_dna_pcc": obs_a,
        "pooled_fusion_pcc": obs_b,
        "pooled_delta_pcc": obs,
        "bootstrap_replicates": int(n_boot),
        "bootstrap_seed": int(seed),
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_positive_fraction": float(np.mean(boot > 0)),
        "randomization_replicates": int(n_perm),
        "randomization_seed": int(seed + 1),
        "randomization_p_two_sided": p_two,
        "randomization_p_gain_one_sided": p_gain,
    }


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = pvals[order]
    m = len(pvals)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--n-randomization", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    fold_all = []
    manifest_all = []
    analyses = list(TRAITS.items()) + [("Stickleback_A_Freq.Lymph", None), ("Drosophila_StarvationRes_M", "__DROSOPHILA__")]
    for i, (trait, directory) in enumerate(analyses):
        if directory is None:
            oof, folds, manifest = load_fish(args.source_root)
        elif directory == "__DROSOPHILA__":
            oof, folds, manifest = load_drosophila(args.source_root)
        else:
            oof, folds, manifest = load_trait(args.source_root, directory)
        stats = paired_inference(
            oof,
            n_boot=args.n_bootstrap,
            n_perm=args.n_randomization,
            seed=args.seed + i * 100,
        )
        stats.update(
            {
                "trait": trait,
                "mean_fold_dna_pcc": float(folds["dna_pcc"].mean()),
                "sd_fold_dna_pcc": float(folds["dna_pcc"].std(ddof=1)),
                "mean_fold_fusion_pcc": float(folds["fusion_pcc"].mean()),
                "sd_fold_fusion_pcc": float(folds["fusion_pcc"].std(ddof=1)),
                "mean_fold_delta_pcc": float(folds["delta_pcc"].mean()),
                "positive_outer_folds": int((folds["delta_pcc"] > 0).sum()),
                "zero_outer_folds": int((folds["delta_pcc"] == 0).sum()),
            }
        )
        summary_rows.append(stats)
        folds.insert(0, "trait", trait)
        fold_all.append(folds)
        for row in manifest:
            row["trait"] = trait
            manifest_all.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["randomization_q_two_sided_BH"] = bh_adjust(
        summary["randomization_p_two_sided"].to_numpy()
    )
    summary["randomization_q_gain_one_sided_BH"] = bh_adjust(
        summary["randomization_p_gain_one_sided"].to_numpy()
    )
    summary.to_csv(args.output_dir / "trace_rna_paired_statistics.csv", index=False)
    pd.concat(fold_all, ignore_index=True).to_csv(
        args.output_dir / "trace_rna_fold_metrics_recomputed.csv", index=False
    )
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "independent_unit": "one accession/line represented by one outer-test OOF row",
                "bootstrap": "paired nonparametric bootstrap of OOF rows; percentile 95% CI",
                "randomization": "within-row exchange of DNA-only and DNA+RNA predictions",
                "multiple_testing": "Benjamini-Hochberg across all seven prespecified completed traits",
                "n_bootstrap": args.n_bootstrap,
                "n_randomization": args.n_randomization,
                "base_seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

