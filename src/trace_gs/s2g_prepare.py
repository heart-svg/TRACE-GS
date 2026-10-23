from pathlib import Path
from biodem import OptimizeLitdataNCV

BASE = Path("/path/to/trace_gs_data")
DATA = BASE / "data/arabidopsis_full/data_prep"
OUT = BASE / "results/arabidopsis_full/S2G/FT10"


def main():
    opt = OptimizeLitdataNCV(
        paths_omics={"gv": str(DATA / "arabidopsis_645_maf05_snp.pkl.gz")},
        path_label=str(DATA / "phenotypes.csv"),
        output_dir=str(OUT / "litdata"),
        # 645 unique accessions are exactly divisible by 5*3=15.
        # This prevents biodem from resampling extra rows to fill fragments.
        k_outer=5,
        k_inner=3,
        which_outer_inner=[0, 0],
        col2use_in_labels=["FT10"],
    )
    opt.run_optimization()
    split_dir = OUT / "litdata/ncv_test_0_val_0"
    splits = {}
    for name in ("trn", "val", "tst"):
        path = split_dir / f"sample_ids_{name}.csv"
        ids = [line.strip() for line in path.read_text().splitlines()[1:] if line.strip()]
        assert len(ids) == len(set(ids)), f"duplicate accession within {name}"
        splits[name] = set(ids)
    assert not (splits["trn"] & splits["val"]), "train/val leakage"
    assert not (splits["trn"] & splits["tst"]), "train/test leakage"
    assert not (splits["val"] & splits["tst"]), "val/test leakage"
    assert len(set().union(*splits.values())) == 645, "expected 645 unique accessions"
    print({name: len(ids) for name, ids in splits.items()}, flush=True)
    print("FULL FT10 S2G PREP DONE", flush=True)


if __name__ == "__main__":
    main()

