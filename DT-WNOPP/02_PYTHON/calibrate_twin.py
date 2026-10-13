
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.home() / "Documents" / "DT-WNOPP"
DATA = ROOT / "03_DATA"
RESULTS = ROOT / "04_RESULTS"
RESULTS.mkdir(parents=True, exist_ok=True)

data_path = DATA / "processed_measurements.csv"
twin_path = RESULTS / "twin_outputs.csv"

data = pd.read_csv(data_path)
twin = pd.read_csv(twin_path)

# Match rows by sequence index only after verifying equal length
# and that both files retain the original processing order.
if len(data) != len(twin):
    raise ValueError(
        f"Row mismatch: measurements={len(data)}, twin={len(twin)}. "
        "Regenerate twin outputs from the current processed CSV."
    )

if not np.allclose(
    data["time_step"].to_numpy(dtype=float),
    twin["simulation_time"].to_numpy(dtype=float),
):
    raise ValueError(
        "Time-step alignment failed. Regenerate twin outputs."
    )

target = "throughput_observed_source_units"
proxy = "throughput_proxy_normalised"

if target not in data or proxy not in twin:
    raise ValueError("Required measured or simulated throughput column missing.")

df = data.copy()
df["twin_proxy"] = twin[proxy].to_numpy(dtype=float)
df["target"] = pd.to_numeric(df[target], errors="coerce")
df["config_group"] = (
    df["scenario"].astype(str)
    .str.replace(r"_rep\d+$", "", regex=True, case=False)
)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["twin_proxy", "target", "config_group"]).reset_index(drop=True)

if df["config_group"].nunique() < 5:
    raise ValueError("Not enough configuration groups for evaluation.")

X = df[["twin_proxy"]]
y = df["target"]
groups = df["config_group"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Fit the calibration mapping only on training configurations.
calibrator = Ridge(alpha=1.0)
calibrator.fit(X_train, y_train)
pred = calibrator.predict(X_test)

# Baseline: training-set mean.
baseline = np.full(len(y_test), y_train.mean())

def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

scores = pd.DataFrame({
    "Training-mean baseline": metrics(y_test, baseline),
    "Calibrated twin": metrics(y_test, pred),
}).T

comparison = pd.DataFrame({
    "scenario": df.iloc[test_idx]["scenario"].to_numpy(),
    "measured_throughput": y_test.to_numpy(),
    "calibrated_twin_prediction": pred,
    "training_mean_prediction": baseline,
    "twin_proxy_normalised": X_test["twin_proxy"].to_numpy(),
})

scores.to_csv(RESULTS / "twin_calibration_metrics.csv")
comparison.to_csv(
    RESULTS / "twin_calibration_predictions.csv",
    index=False,
)

print("\nCALIBRATION COEFFICIENT:", calibrator.coef_[0])
print("CALIBRATION INTERCEPT:", calibrator.intercept_)
print("\nHELD-OUT CALIBRATION RESULTS")
print(scores.round(4).to_string())
print("\nSaved calibration metrics and predictions in:", RESULTS)
