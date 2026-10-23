from pathlib import Path
import hashlib
import ast
import pandas as pd

root = Path(__file__).resolve().parents[2]
oof = pd.read_csv(root / "results/oof/seven_trait_formal_oof_predictions_long.csv", dtype={"ID": str})
assert oof["dataset"].nunique() == 7, oof["dataset"].unique()
assert set(oof["outer"].dropna().astype(int)) == {0, 1, 2, 3, 4}
assert not oof[["dataset", "model", "ID"]].duplicated().any()

for dataset, g in oof.groupby("dataset"):
    n = g.groupby("model")["ID"].nunique()
    assert n.nunique() == 1, (dataset, n.to_dict())
    fold = g.groupby("ID")["outer"].nunique()
    assert (fold == 1).all(), dataset

zero = [p for p in root.rglob("*") if p.is_file() and p.stat().st_size == 0]
assert not zero, zero
for script in root.rglob("*.py"):
    ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))
print(f"PASS: 7 datasets, {oof['model'].nunique()} model labels, {len(oof):,} OOF rows")

