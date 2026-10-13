
from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.home() / "Documents" / "DT-WNOPP"
DATA = ROOT / "03_DATA"
RESULTS = ROOT / "04_RESULTS"
RESULTS.mkdir(parents=True, exist_ok=True)

INPUT = DATA / "processed_measurements.csv"
df = pd.read_csv(INPUT)

target = "throughput_observed_source_units"

required = [
    target,
    "scenario",
    "traffic_load",
    "resource_fraction",
    "offered_downlink_source_units",
    "bandwidth_source_units",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=[target, "scenario"]).copy()
df = df.replace([np.inf, -np.inf], np.nan)
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).reset_index(drop=True)

# Extract physical/configuration information from scenario labels.
# Scenario names are parsed into features; the full name is not a feature.
scenario = df["scenario"].astype(str)

def extract_number(pattern):
    extracted = scenario.str.extract(pattern, flags=re.IGNORECASE)[0]
    return pd.to_numeric(extracted, errors="coerce")

df["config_bandwidth_mhz"] = extract_number(r"_(\d+(?:\.\d+)?)mhz")
df["config_slots"] = extract_number(r"_(\d+)slots")
df["config_ratio"] = extract_number(r"_ratio(\d+)")
df["uplink_share_pct"] = extract_number(r"_(\d+)up_\d+down")
df["downlink_share_pct"] = extract_number(r"_\d+up_(\d+)down")

# Independent runs of the same configuration must remain in one split.
# Example: rep1, rep2, rep3 of the same configuration share a group.
df["config_group"] = scenario.str.replace(
    r"_rep\d+$", "", regex=True, case=False
)

candidate_features = [
    "traffic_load",
    "resource_fraction",
    "offered_downlink_source_units",
    "bandwidth_source_units",
    "config_bandwidth_mhz",
    "config_slots",
    "config_ratio",
    "uplink_share_pct",
    "downlink_share_pct",
]

features = [
    c for c in candidate_features
    if c in df.columns
    and df[c].notna().any()
    and df[c].nunique(dropna=True) > 1
]

if len(df) < 30:
    raise ValueError("Too few observations for a useful holdout evaluation.")

if df["config_group"].nunique() < 5:
    raise ValueError(
        "Too few distinct configuration groups for group-based evaluation."
    )

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df[target].astype(float)
groups = df["config_group"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

def calculate_metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }

models = {
    "Training-mean baseline": None,
    "Ridge": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

score_rows = []
prediction_data = pd.DataFrame({
    "row_index": test_idx,
    "scenario": df.iloc[test_idx]["scenario"].to_numpy(),
    "actual_throughput": y_test.to_numpy(),
})

for name, model in models.items():
    if model is None:
        pred = np.full(len(y_test), y_train.mean())
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    row = {"Model": name}
    row.update(calculate_metrics(y_test, pred))
    score_rows.append(row)

    column_name = (
        "training_mean_prediction"
        if model is None else name.lower().replace(" ", "_") + "_prediction"
    )
    prediction_data[column_name] = pred

metrics = pd.DataFrame(score_rows).set_index("Model")
metrics.to_csv(RESULTS / "throughput_model_metrics_v2.csv")
prediction_data.to_csv(
    RESULTS / "throughput_predictions_v2.csv",
    index=False,
)

print("\nFEATURES USED:")
print(features)

print("\nTRAINING ROWS:", len(train_idx))
print("TEST ROWS:", len(test_idx))
print("TRAIN CONFIGURATIONS:", groups.iloc[train_idx].nunique())
print("TEST CONFIGURATIONS:", groups.iloc[test_idx].nunique())

print("\nHELD-OUT PERFORMANCE")
print(metrics.round(4).to_string())

print("\nSaved:")
print(RESULTS / "throughput_model_metrics_v2.csv")
print(RESULTS / "throughput_predictions_v2.csv")
