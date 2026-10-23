from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
oof_path = root / "results/oof/seven_trait_formal_oof_predictions_long.csv"
d = pd.read_csv(oof_path, dtype={"ID": str})
if "source_file" in d:
    d["source_file"] = ("source_snapshot/" + d["dataset"].astype(str) + "/" +
                        d["model"].astype(str) + "/oof_predictions.csv")
d.to_csv(oof_path, index=False)

for path in (root / "results/figure_source_data").glob("*.csv"):
    x = pd.read_csv(path)
    for column in x.columns:
        if column.lower() in {"source", "source_file", "source_path"}:
            x[column] = x[column].astype(str).map(
                lambda value: "source_snapshot/" + Path(value.replace("\\", "/")).name
                if value and value.lower() != "nan" else value
            )
    x.to_csv(path, index=False)
