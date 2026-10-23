"""Audit and freeze the formal five-outer-fold evaluation manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

TRAITS = ["FT10", "FT16", "Diameter", "rSeeds_thi"]


def read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    column = "ID" if "ID" in frame.columns else frame.columns[0]
    ids = frame[column].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise AssertionError(f"Duplicate accessions in {path}")
    return ids


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    phenotype = pd.read_csv(args.split_root / "phenotypes_guard4_common_unique.csv")
    phenotype["ID"] = phenotype.ID.astype(str)
    if phenotype.ID.duplicated().any(): raise AssertionError("Phenotype accessions are not unique")
    if phenotype[TRAITS].isna().any().any(): raise AssertionError("Missing formal phenotypes")
    universe = set(phenotype.ID)
    rows, outer_tests = [], []
    for outer in range(5):
        expected_test = None
        for inner in range(3):
            root = args.split_root / f"outer_{outer}" / f"inner_{inner}"
            train, val, test = (read_ids(root/f"{name}_ids.csv") for name in ("train", "val", "test"))
            st, sv, ss = set(train), set(val), set(test)
            if st&sv or st&ss or sv&ss: raise AssertionError(f"Leakage in outer={outer}, inner={inner}")
            if st|sv|ss != universe: raise AssertionError(f"Incomplete universe outer={outer}, inner={inner}")
            if expected_test is None: expected_test = test
            elif set(expected_test) != ss: raise AssertionError(f"Outer test changed across inner folds: {outer}")
            rows.append({"outer":outer,"inner":inner,"n_train":len(train),"n_val":len(val),"n_test":len(test),
                         "train_sha256":digest(train),"val_sha256":digest(val),"test_sha256":digest(test)})
        outer_tests.append(set(expected_test))
    for i in range(5):
        for j in range(i+1,5):
            if outer_tests[i]&outer_tests[j]: raise AssertionError(f"Outer test overlap: {i},{j}")
    if set().union(*outer_tests) != universe: raise AssertionError("Outer tests do not cover universe exactly")
    pd.DataFrame(rows).to_csv(args.output/"formal_cv_manifest.csv",index=False)
    metadata={"status":"FROZEN","n_unique":len(universe),"outer_folds":5,"inner_folds":3,
              "model_seeds":[20260825,20260826,20260827,20260828,20260829],
              "traits":TRAITS,"fold0_development_excluded":True,
              "outer_tests_disjoint_and_complete":True,
              "required_order":["fold-specific S2G","DNA-only","OOF RNA residual fusion","DEM","DEM-Fix"]}
    (args.output/"formal_cv_protocol.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__": main()

