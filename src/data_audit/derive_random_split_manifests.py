from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
src = root / "results/oof/seven_trait_formal_oof_predictions_long.csv"
out = root / "splits/random_fivefold"
out.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(src, dtype={"ID": str})
for dataset, g in d.groupby("dataset", sort=True):
    counts = g.groupby("ID")["outer"].nunique()
    if not (counts == 1).all():
        raise RuntimeError(f"Inconsistent outer folds across models: {dataset}")
    manifest = (g[["species", "trait", "dataset", "ID", "outer"]]
                .drop_duplicates()
                .sort_values(["outer", "ID"]))
    if manifest["ID"].duplicated().any():
        raise RuntimeError(f"Duplicate material IDs: {dataset}")
    manifest.to_csv(out / f"{dataset}_outer_test_manifest.csv", index=False)

