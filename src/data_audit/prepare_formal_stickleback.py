"""Prepare the public stickleback genotype/RNA/phenotype data for TRACE-GS.

The script is intentionally phenotype-blind when assigning folds. Families are
kept intact in both outer and inner CV. It writes an ancestry-coded VCF solely
as an interoperable genotype container; REF/ALT labels are placeholders and
must not be interpreted as nucleotide alleles.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/path/to/trace_gs_data/data/stickleback")
RAW = ROOT / "raw"
OUT = ROOT / "formal"
SEED = 20261009


def balanced_group_assignment(meta: pd.DataFrame, group: str, strata: str, k: int, seed: int):
    rng = np.random.default_rng(seed)
    tab = pd.crosstab(meta[group], meta[strata])
    sizes = tab.sum(axis=1)
    jitter = pd.Series(rng.random(len(tab)), index=tab.index)
    order = sorted(tab.index, key=lambda g: (-int(sizes[g]), float(jitter[g])))
    fold_n = np.zeros(k, dtype=int)
    fold_s = np.zeros((k, tab.shape[1]), dtype=int)
    assignment = {}
    target_n = len(meta) / k
    target_s = tab.sum(axis=0).to_numpy() / k
    for g in order:
        vec = tab.loc[g].to_numpy(dtype=int)
        n = int(vec.sum())
        scores = []
        for f in range(k):
            new_n = fold_n.copy(); new_n[f] += n
            new_s = fold_s.copy(); new_s[f] += vec
            score = np.sum((new_n - target_n) ** 2) / max(target_n**2, 1)
            score += np.sum((new_s - target_s) ** 2 / np.maximum(target_s, 1) ** 2)
            scores.append(score)
        best = int(np.argmin(scores))
        assignment[g] = best
        fold_n[best] += n
        fold_s[best] += vec
    return assignment


def md5(path: Path):
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(RAW / "MergedTagseqQTL_GENOCORRECTED.csv", low_memory=False)
    pos = pd.read_csv(RAW / "QTL_snp_pos_corrected.csv")
    txmap = pd.read_csv(RAW / "MapENSGACT_ENSGACG.csv")

    assert len(merged) == 351 and merged["sample_ID"].nunique() == 351
    genotype_cols = list(merged.columns[112:346])
    expression_cols = [c for c in merged.columns if c.startswith("ENSGACT")]
    assert len(genotype_cols) == len(pos) == 234
    assert len(expression_cols) == 25845

    ids = merged["sample_ID"].astype(str).to_numpy()
    meta_cols = [
        "sample_ID", "ID", "uniquefam", "Family", "F", "M", "CrossDir",
        "sex", "WormSource", "worm_present", "infected", "date", "DPE",
        "weight", "length", "HK_count", "n_worms_found", "avg_worm_mass",
        "total_worm_mass", "max_worm_mass", "A_Freq.Gran", "A_Freq.Lymph",
        "BG_ROS_L", "BG_ROS_G", "PMA_ROS_L", "PMA_ROS_G",
    ]
    meta = merged[meta_cols].copy().rename(columns={"sample_ID": "sample_id", "sex": "Sex"})
    meta.to_csv(OUT / "sample_phenotype_covariate_manifest.csv", index=False)

    # Actual observed marker genotypes encoded as dosage of parental B ancestry.
    dosage_all = merged[genotype_cols].replace({"AA": 0, "AB": 1, "BB": 2, "-": np.nan})
    dosage_all.columns = [f"m{i+1:03d}" for i in range(len(genotype_cols))]
    positioned = pos["Pos"].notna() & pos["LGn"].notna()
    dosage = dosage_all.loc[:, positioned.to_numpy()].copy()
    dosage.insert(0, "sample_id", ids)
    dosage.to_csv(OUT / "genotype_dosage.csv.gz", index=False, compression="gzip")

    marker = pos.loc[positioned].reset_index(drop=True).copy()
    marker.insert(0, "marker_id", dosage.columns[1:])
    marker["source_column"] = np.asarray(genotype_cols)[positioned.to_numpy()]
    marker.to_csv(OUT / "marker_map.csv", index=False)

    # Standards-compatible container. A/T are declared ancestry-code placeholders.
    vcf_path = OUT / "stickleback_351_ancestry_coded.vcf.gz"
    with gzip.open(vcf_path, "wt", newline="") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##source=Bolnick_stickleback_ddRAD_parental_ancestry_markers\n")
        fh.write("##reference=gasAcu1_BROADS1_coordinates\n")
        fh.write("##INFO=<ID=ANCESTRY_CODED,Number=0,Type=Flag,Description=REF/ALT are placeholder parental ancestry labels, not nucleotide alleles>\n")
        fh.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>\n")
        fh.write("##FORMAT=<ID=DS,Number=1,Type=Float,Description=Parental B ancestry dosage>\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ids) + "\n")
        for j, row in marker.iterrows():
            vals = dosage.iloc[:, j + 1].to_numpy()
            calls = []
            for x in vals:
                if not np.isfinite(x): calls.append("./.:.")
                elif int(x) == 0: calls.append("0/0:0")
                elif int(x) == 1: calls.append("0/1:1")
                else: calls.append("1/1:2")
            fh.write(f"{int(row.LGn)}\t{int(row.Pos)}\t{row.marker_id}\tA\tT\t.\tPASS\tANCESTRY_CODED\tGT:DS\t" + "\t".join(calls) + "\n")

    # Aggregate transcript counts to Ensembl genes using the supplied assembly-matched map.
    expr = merged[expression_cols].copy()
    base_tx = pd.Index([c.split(".")[0] for c in expression_cols])
    tx_to_gene = txmap.dropna(subset=["ENSGACT", "ENSGACG"]).drop_duplicates("ENSGACT").set_index("ENSGACT")["ENSGACG"]
    mapped_gene = base_tx.map(tx_to_gene)
    keep = mapped_gene.notna()
    expr = expr.loc[:, keep]
    expr.columns = mapped_gene[keep]
    gene_expr = expr.T.groupby(level=0).sum().T
    gene_expr.insert(0, "sample_id", ids)
    gene_expr.to_csv(OUT / "rna_gene_counts.csv.gz", index=False, compression="gzip")

    gene_bed = (txmap.dropna(subset=["ENSGACG", "LG", "Start", "Stop"])
                .groupby(["ENSGACG", "LG"], as_index=False)
                .agg(start=("Start", "min"), end=("Stop", "max")))
    gene_bed["LG"] = gene_bed["LG"].astype(int)
    gene_bed[["LG", "start", "end", "ENSGACG"]].to_csv(
        OUT / "gasAcu1_gene_regions.bed", sep="\t", header=False, index=False
    )

    # Phenotype-blind family-grouped 5x3 nested CV.
    split_meta = meta[["sample_id", "uniquefam", "CrossDir"]].copy()
    outer_assign = balanced_group_assignment(split_meta, "uniquefam", "CrossDir", 5, SEED)
    split_meta["outer_test_fold"] = split_meta["uniquefam"].map(outer_assign)
    rows = []
    for outer in range(5):
        train_meta = split_meta[split_meta.outer_test_fold != outer].copy()
        inner_assign = balanced_group_assignment(train_meta, "uniquefam", "CrossDir", 3, SEED + outer + 1)
        for _, r in split_meta.iterrows():
            rows.append({
                "sample_id": r.sample_id,
                "family_id": int(r.uniquefam),
                "cross_type": r.CrossDir,
                "outer": outer,
                "partition": "test" if r.outer_test_fold == outer else "outer_train",
                "inner_holdout": "" if r.outer_test_fold == outer else inner_assign[r.uniquefam],
            })
    splits = pd.DataFrame(rows)
    splits.to_csv(OUT / "nested_5x3_family_grouped_splits.csv", index=False)

    # Fail-closed split and modality audit.
    for outer in range(5):
        s = splits[splits.outer == outer]
        tf = set(s.loc[s.partition == "test", "family_id"])
        rf = set(s.loc[s.partition == "outer_train", "family_id"])
        assert not (tf & rf)
        assert set(s.loc[s.partition == "outer_train", "inner_holdout"].astype(int)) == {0, 1, 2}
    report = {
        "status": "PASS",
        "source": "public Figshare project 226689",
        "samples": len(ids), "unique_sample_ids": len(set(ids)), "families": int(meta.uniquefam.nunique()),
        "cross_counts": meta.CrossDir.value_counts().to_dict(),
        "markers_raw": len(genotype_cols), "markers_positioned_used": int(positioned.sum()),
        "markers_unpositioned_excluded": int((~positioned).sum()),
        "marker_missing_rate": float(dosage.iloc[:, 1:].isna().mean().mean()),
        "transcripts": len(expression_cols), "mapped_gene_features": gene_expr.shape[1] - 1,
        "vcf_notice": "A/T are ancestry-code placeholders; positions and individual genotype calls are observed ddRAD marker data.",
        "split_protocol": "family-grouped 5 outer x 3 inner, phenotype-blind",
        "files": {},
    }
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "FORMAL_DATA_AUDIT.json":
            report["files"][path.name] = {"bytes": path.stat().st_size, "md5": md5(path)}
    with (OUT / "FORMAL_DATA_AUDIT.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

