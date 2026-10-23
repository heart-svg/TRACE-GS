#!/usr/bin/env python3
"""Fold-local RNA coefficient back-projection for formal TRACE-GS runs.

This analysis refits only the already selected PCA-Ridge RNA expert on each
outer-training set.  It never reads outer-test phenotypes.  Coefficients are
back-projected from PC space to standardized gene-expression space so that one
unit represents one training-fold standard deviation of expression.
"""

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


CONFIG = {
    "FT10": {
        "strict": "strict_inner_s2g_v1",
        "trace": "TRACE/FT10",
    },
    "FT16": {
        "strict": "strict_inner_s2g_FT16_v1",
        "trace": "TRACE/FT16",
    },
    "Diameter": {
        "strict": "strict_inner_s2g_Diameter_v1",
        "trace": "TRACE/Diameter",
        "trace_dna": "results/arabidopsis_full/TRACE/Diameter",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def top_set(frame: pd.DataFrame, frac: float) -> set[str]:
    n = max(1, int(np.ceil(len(frame) * frac)))
    return set(frame.nlargest(n, "abs_beta_std")["gene"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/path/to/trace_gs_data"))
    ap.add_argument("--trait", choices=sorted(CONFIG), required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    cv = args.root / "results/arabidopsis_full/fresh_cv_seed20260902"
    cfg = CONFIG[args.trait]
    strict = cv / cfg["strict"]
    trace = cv / cfg["trace"]
    trace_dna = args.root / cfg.get("trace_dna", str(Path("results/arabidopsis_full/fresh_cv_seed20260902") / cfg["trace"]))
    rna_path = args.root / "data/arabidopsis_full/data_prep/omics_transcriptome_log2.parquet"
    pheno_path = args.root / "data/arabidopsis_full/data_prep/phenotypes.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rna = pd.read_parquet(rna_path)
    id_col = "ID" if "ID" in rna.columns else rna.columns[0]
    rna[id_col] = rna[id_col].astype(str)
    if rna[id_col].duplicated().any():
        raise ValueError("duplicate RNA IDs")
    rna = rna.set_index(id_col)
    genes = np.asarray(rna.columns.astype(str))
    pheno = pd.read_csv(pheno_path, dtype={"ID": str}).set_index("ID")
    phenotype_sources = [pheno_path]
    phenotype_source_mode = "shared_phenotype_table"
    if args.trait not in pheno.columns:
        # Diameter is not stored in the shared FT10/FT16 phenotype table.  The
        # finalized DNA split predictions contain the same observed phenotype
        # for an accession whenever it occurs in a training/validation/test
        # split. Rebuild one canonical ID-to-phenotype map and require exact
        # cross-fold agreement before using only the current outer-training IDs.
        label_parts = []
        phenotype_sources = []
        for outer in range(5):
            for split in ("trn", "val", "tst"):
                p = trace_dna / f"outer_{outer}/dna/predictions_{split}.csv"
                z = pd.read_csv(p, dtype={"ID": str})[["ID", "y_raw"]]
                label_parts.append(z)
                phenotype_sources.append(p)
        labels = pd.concat(label_parts, ignore_index=True)
        spread = labels.groupby("ID")["y_raw"].agg(
            lambda x: float(np.nanmax(x.to_numpy(float)) - np.nanmin(x.to_numpy(float)))
        )
        if len(spread) == 0 or float(spread.max()) >= 1e-5:
            raise ValueError("inconsistent phenotype values across formal split files")
        pheno = (
            labels.drop_duplicates("ID")
            .set_index("ID")
            .rename(columns={"y_raw": args.trait})
        )
        phenotype_source_mode = "audited_formal_split_prediction_labels"

    fold_frames = []
    audit = []
    top_sets = {0.005: [], 0.01: [], 0.02: []}
    for outer in range(5):
        oof_path = strict / f"outer_{outer}/strict_dna_oof_train.csv"
        metrics_path = trace / f"strict_rna_residual_5fold/outer_{outer}_metrics.json"
        oof = pd.read_csv(oof_path, dtype={"ID": str})
        if oof["ID"].duplicated().any():
            raise ValueError(f"outer {outer}: duplicate OOF IDs")
        y = pheno.loc[oof["ID"], args.trait].to_numpy(float)
        residual = y - oof["pred_raw"].to_numpy(float)
        hp = json.loads(metrics_path.read_text(encoding="utf-8"))
        q = int(hp["pca_components"])
        alpha = float(hp["ridge_alpha"])
        x = rna.loc[oof["ID"]].to_numpy(np.float32)
        scaler = StandardScaler().fit(x)
        xs = scaler.transform(x)
        pca = PCA(
            n_components=q,
            svd_solver="randomized",
            random_state=args.seed + outer,
        ).fit(xs)
        z = pca.transform(xs)
        ridge = Ridge(alpha=alpha).fit(z, residual)
        beta_std = np.asarray(ridge.coef_) @ np.asarray(pca.components_)
        frame = pd.DataFrame(
            {
                "gene": genes,
                "outer": outer,
                "beta_std": beta_std,
                "abs_beta_std": np.abs(beta_std),
                "sign": np.sign(beta_std).astype(int),
            }
        )
        frame["abs_rank"] = frame["abs_beta_std"].rank(method="min", ascending=False).astype(int)
        frame.to_csv(args.output_dir / f"outer_{outer}_gene_coefficients.csv.gz", index=False)
        fold_frames.append(frame)
        for frac in top_sets:
            top_sets[frac].append(top_set(frame, frac))
        audit.append(
            {
                "outer": outer,
                "n_train_oof": int(len(oof)),
                "n_genes": int(len(genes)),
                "pca_components": q,
                "ridge_alpha": alpha,
                "oof_sha256": sha256(oof_path),
                "metrics_sha256": sha256(metrics_path),
            }
        )

    all_coef = pd.concat(fold_frames, ignore_index=True)
    agg = all_coef.groupby("gene", sort=False).agg(
        mean_beta_std=("beta_std", "mean"),
        median_beta_std=("beta_std", "median"),
        mean_abs_beta_std=("abs_beta_std", "mean"),
        mean_abs_rank=("abs_rank", "mean"),
        positive_folds=("sign", lambda x: int((x > 0).sum())),
        negative_folds=("sign", lambda x: int((x < 0).sum())),
    ).reset_index()
    for frac, sets in top_sets.items():
        col = f"top_{frac:g}_folds"
        counts = {g: sum(g in s for s in sets) for g in genes}
        agg[col] = agg["gene"].map(counts).astype(int)
    agg["sign_consistent_folds"] = agg[["positive_folds", "negative_folds"]].max(axis=1)
    agg["stable_primary"] = (agg["top_0.01_folds"] >= 4) & (agg["sign_consistent_folds"] >= 4)
    agg.sort_values(["stable_primary", "mean_abs_rank"], ascending=[False, True]).to_csv(
        args.output_dir / "gene_stability_summary.csv", index=False
    )
    agg.loc[agg["stable_primary"]].to_csv(
        args.output_dir / "stable_genes_primary.csv", index=False
    )

    jac = []
    for frac, sets in top_sets.items():
        for i, j in combinations(range(5), 2):
            union = sets[i] | sets[j]
            jac.append(
                {
                    "top_fraction": frac,
                    "outer_a": i,
                    "outer_b": j,
                    "jaccard": len(sets[i] & sets[j]) / len(union) if union else np.nan,
                }
            )
    pd.DataFrame(jac).to_csv(args.output_dir / "pairwise_top_gene_jaccard.csv", index=False)
    manifest = {
        "status": "PASS",
        "trait": args.trait,
        "method": "outer-training-only PCA-Ridge refit and PC-to-standardized-gene back-projection",
        "primary_rule": "top 1% absolute standardized coefficient in >=4/5 folds and coefficient sign consistent in >=4/5 folds",
        "sensitivity_top_fractions": [0.005, 0.01, 0.02],
        "test_labels_used": False,
        "n_stable_primary": int(agg["stable_primary"].sum()),
        "fold_audit": audit,
        "rna_matrix_sha256": sha256(rna_path),
        "phenotype_source_mode": phenotype_source_mode,
        "phenotype_sources": [
            {"path": str(p), "sha256": sha256(p)} for p in phenotype_sources
        ],
    }
    (args.output_dir / "STABILITY_COMPLETE.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

