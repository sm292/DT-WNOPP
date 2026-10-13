
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Documents" / "DT-WNOPP"
DATA = ROOT / "03_DATA"
RESULTS = ROOT / "04_RESULTS"

files = {
    "processed_data": DATA / "processed_measurements.csv",
    "twin_outputs": RESULTS / "twin_outputs.csv",
    "prediction_metrics": RESULTS / "throughput_model_metrics.csv",
    "predictions": RESULTS / "throughput_predictions.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 65}")
    print(name.upper(), "-", path)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("\nNumeric summary:")
        print(numeric.describe().round(4).to_string())

        print("\nMissing values:")
        print(numeric.isna().sum().to_string())

        print("\nInfinite values:")
        print(
            pd.Series(
                np.isinf(numeric.to_numpy()).sum(axis=0),
                index=numeric.columns,
            ).to_string()
        )

    if "scenario" in df.columns:
        print("\nRows per scenario:")
        print(df["scenario"].value_counts().head(15).to_string())

print("\nAudit completed.")
