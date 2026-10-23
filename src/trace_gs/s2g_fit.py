from pathlib import Path
from biodem import SNP2GBFitPipe

BASE = Path("/path/to/trace_gs_data")
OUT = BASE / "results/arabidopsis_full/S2G/FT10"


def main():
    pipe = SNP2GBFitPipe(
        litdata_dir=str(OUT / "litdata"),
        list_ncv=[[0, 0]],
        log_dir=str(OUT / "models"),
        regression=True,
        n_jobs=1,
        n_trials=2,
    )
    pipe.train_pipeline()
    print("FULL FT10 S2G FIT DONE", flush=True)


if __name__ == "__main__":
    main()

