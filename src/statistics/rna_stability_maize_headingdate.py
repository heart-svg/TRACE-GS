#!/usr/bin/env python3
"""Training-fold-only RNA coefficient stability for maize Headingdate."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/path/to/trace_gs_data/maizea"),
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261014)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace = args.root / "results/TRACE/Headingdate"
    inner = trace / "strict_inner_oof"
    metrics = trace / "strict_rna_residual_5fold"
    rna_path = args.root / "data_prep/omics_transcriptome.parquet"
    rna = pd.read_parquet(rna_path)
    if "ID" not in rna.columns:
        rna = rna.reset_index()
    rna["ID"] = rna["ID"].astype(str)
    if rna["ID"].duplicated().any():
        raise ValueError("duplicate RNA IDs")
    rna = rna.set_index("ID")
    genes = np.asarray(rna.columns.astype(str))

    frames = []
    fold_sets = {0.005: [], 0.01: [], 0.02: []}
    audit = []
    for outer in range(5):
        parts = []
        part_hashes = []
        for inner_fold in range(3):
            p = inner / f"outer_{outer}/inner_{inner_fold}/predictions_val.csv"
            z = pd.read_csv(p, dtype={"ID": str})[["ID", "y_raw", "pred_raw"]]
            z["inner"] = inner_fold
            parts.append(z)
            part_hashes.append({"inner": inner_fold, "sha256": digest(p), "n": len(z)})
        oof = pd.concat(parts, ignore_index=True)
        if oof["ID"].duplicated().any():
            raise ValueError(f"outer {outer}: duplicated inner holdout IDs")
        hp_path = metrics / f"outer_{outer}_metrics.json"
        hp = json.loads(hp_path.read_text(encoding="utf-8"))
        q, alpha = int(hp["pca_components"]), float(hp["ridge_alpha"])
        residual = oof["y_raw"].to_numpy(float) - oof["pred_raw"].to_numpy(float)
        x = rna.loc[oof["ID"]].to_numpy(np.float32)
        scaler = StandardScaler().fit(x)
        xs = scaler.transform(x)
        pca = PCA(n_components=q, svd_solver="randomized", random_state=args.seed + outer).fit(xs)
        ridge = Ridge(alpha=alpha).fit(pca.transform(xs), residual)
        beta = np.asarray(ridge.coef_) @ np.asarray(pca.components_)
        f = pd.DataFrame(
            {
                "gene": genes,
                "outer": outer,
                "beta_std": beta,
                "abs_beta_std": np.abs(beta),
                "sign": np.sign(beta).astype(int),
            }
        )
        f["abs_rank"] = f["abs_beta_std"].rank(method="min", ascending=False).astype(int)
        f.to_csv(args.output_dir / f"outer_{outer}_gene_coefficients.csv.gz", index=False)
        frames.append(f)
        for frac in fold_sets:
            n = max(1, int(np.ceil(len(f) * frac)))
            fold_sets[frac].append(set(f.nlargest(n, "abs_beta_std")["gene"]))
        audit.append(
            {
                "outer": outer,
                "n_outer_train_oof": int(len(oof)),
                "pca_components": q,
                "ridge_alpha": alpha,
                "inner_holdout_sources": part_hashes,
                "metrics_sha256": digest(hp_path),
            }
        )

    coef = pd.concat(frames, ignore_index=True)
    agg = coef.groupby("gene", sort=False).agg(
        mean_beta_std=("beta_std", "mean"),
        median_beta_std=("beta_std", "median"),
        mean_abs_beta_std=("abs_beta_std", "mean"),
        mean_abs_rank=("abs_rank", "mean"),
        positive_folds=("sign", lambda x: int((x > 0).sum())),
        negative_folds=("sign", lambda x: int((x < 0).sum())),
    ).reset_index()
    for frac, sets in fold_sets.items():
        agg[f"top_{frac:g}_folds"] = agg["gene"].map(
            {g: sum(g in s for s in sets) for g in genes}
        ).astype(int)
    agg["sign_consistent_folds"] = agg[["positive_folds", "negative_folds"]].max(axis=1)
    agg["stable_primary"] = (agg["top_0.01_folds"] >= 4) & (agg["sign_consistent_folds"] >= 4)
    agg.sort_values(["stable_primary", "mean_abs_rank"], ascending=[False, True]).to_csv(
        args.output_dir / "gene_stability_summary.csv", index=False
    )
    agg.loc[agg["stable_primary"]].to_csv(args.output_dir / "stable_genes_primary.csv", index=False)

    jac = []
    for frac, sets in fold_sets.items():
        for i, j in combinations(range(5), 2):
            u = sets[i] | sets[j]
            jac.append(
                {
                    "top_fraction": frac,
                    "outer_a": i,
                    "outer_b": j,
                    "jaccard": len(sets[i] & sets[j]) / len(u) if u else np.nan,
                }
            )
    pd.DataFrame(jac).to_csv(args.output_dir / "pairwise_top_gene_jaccard.csv", index=False)
    report = {
        "status": "PASS",
        "trait": "Headingdate",
        "method": "outer-training inner-holdout DNA residuals; PCA-Ridge back-projected to standardized gene space",
        "analysis_seed": args.seed,
        "primary_rule": "top 1% absolute standardized coefficient in >=4/5 folds and coefficient sign consistent in >=4/5 folds",
        "test_labels_used": False,
        "n_stable_primary": int(agg["stable_primary"].sum()),
        "fold_audit": audit,
        "rna_matrix_sha256": digest(rna_path),
    }
    (args.output_dir / "STABILITY_COMPLETE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

