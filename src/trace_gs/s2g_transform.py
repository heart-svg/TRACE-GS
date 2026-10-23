from pathlib import Path
from biodem import SNP2GBTransPipe

BASE = Path("/path/to/trace_gs_data")
OUT = BASE / "results/arabidopsis_full/S2G/FT10"


def main():
    pipe = SNP2GBTransPipe(
        dir_log=str(OUT / "models"),
        dir_output=str(OUT / "transf"),
    )
    pipe.convert_snp(
        dir_litdata=str(OUT / "litdata"),
        list_ncv=[[0, 0]],
    )
    print("FULL FT10 S2G TRANSFORM DONE", flush=True)


if __name__ == "__main__":
    main()

