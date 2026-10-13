"""Create a row-preserving, multi-objective resource diagnostic.

This reports observed outcomes and exploratory comparisons only. It does not
estimate causal effects of changing resource allocation.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path.home() / "Documents" / "DT-WNOPP"
DATA_PATH = ROOT / "03_DATA" / "processed_measurements.csv"
RESULTS = ROOT / "04_RESULTS"
OUTPUT_PATH = RESULTS / "resource_optimisation_diagnostic_corrected.csv"

METRICS = {
    "throughput_observed_source_units": "max",
    "jitter_observed_ms": "min",
    "packet_loss_observed_source_units": "min",
}
SCORE_COLUMNS = {
    "throughput_observed_source_units": "throughput_desirability",
    "jitter_observed_ms": "jitter_desirability",
    "packet_loss_observed_source_units": "packet_loss_desirability",
}
ANOMALY_Z_THRESHOLD = 3.5


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _robust_anomaly_mask(values: pd.Series) -> pd.Series:
    """Flag extreme values within a repeated, comparable operating condition."""
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    result = pd.Series(False, index=values.index)
    if len(valid) < 4:
        return result

    median = valid.median()
    mad = (valid - median).abs().median()
    if mad > 0:
        score = 0.67448975 * (valid - median).abs() / mad
        result.loc[valid.index] = score > ANOMALY_Z_THRESHOLD
        return result

    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        result.loc[valid.index] = (valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)
    return result


def _normalised_desirability(values: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    low, high = numeric.min(), numeric.max()
    if pd.isna(low) or pd.isna(high):
        return pd.Series(np.nan, index=values.index)
    if high == low:
        return pd.Series(1.0, index=values.index).where(numeric.notna())
    desirability = (numeric - low) / (high - low)
    return desirability if direction == "max" else 1.0 - desirability


def _pareto_front(frame: pd.DataFrame, metric_columns: list[str]) -> pd.Series:
    """Mark non-dominated rows over the three observed outcomes."""
    values = frame[metric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)
    # Convert all objectives to maximisation; each column is range-scaled.
    for i, metric in enumerate(metric_columns):
        column = values[:, i]
        lo, hi = np.nanmin(column), np.nanmax(column)
        values[:, i] = 1.0 if hi == lo else (column - lo) / (hi - lo)
        if METRICS[metric] == "min":
            values[:, i] = 1.0 - values[:, i]

    complete = np.isfinite(values).all(axis=1)
    efficient = np.zeros(len(frame), dtype=bool)
    indexes = np.flatnonzero(complete)
    for i in indexes:
        others = values[indexes]
        dominates = np.all(others >= values[i], axis=1) & np.any(others > values[i], axis=1)
        if not dominates.any():
            efficient[i] = True
    return pd.Series(efficient, index=frame.index)


def build_diagnostic(source: pd.DataFrame) -> pd.DataFrame:
    """Return all source rows with diagnostic columns appended."""
    df = source.copy(deep=True)
    missing = [column for column in METRICS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required outcome columns: {missing}")

    numeric_columns = list(METRICS)
    numeric_columns += [
        name for name in ("resource_fraction", "traffic_load", "time_step")
        if name in df.columns
    ]
    _numeric(df, numeric_columns)

    # Compare repetitions only when their condition identifiers match.
    group_columns = [
        name for name in (
            "source", "traffic_load", "resource_fraction", "bandwidth_source_units",
            "config_bandwidth_mhz", "config_slots", "config_ratio",
            "uplink_share_pct", "downlink_share_pct",
        ) if name in df.columns
    ]
    if "scenario" in df.columns:
        df["_diagnostic_condition"] = (
            df["scenario"].astype("string").str.replace(r"_rep\d+$", "", regex=True)
        )
        group_columns.append("_diagnostic_condition")

    df["anomaly_flag"] = False
    df["anomaly_metrics"] = ""
    if group_columns:
        grouped = df.groupby(group_columns, dropna=False, sort=False)
        for metric in METRICS:
            flagged = grouped[metric].transform(_robust_anomaly_mask)
            df.loc[flagged, "anomaly_flag"] = True
            existing = df.loc[flagged, "anomaly_metrics"]
            df.loc[flagged, "anomaly_metrics"] = existing.map(
                lambda value: f"{value};{metric}".strip(";")
            )

    for metric, direction in METRICS.items():
        df[SCORE_COLUMNS[metric]] = _normalised_desirability(df[metric], direction)

    score_cols = list(SCORE_COLUMNS.values())
    # Equal weights are explicit and transparent; require all three outcomes.
    df["balanced_objective_score"] = df[score_cols].mean(axis=1).where(
        df[score_cols].notna().all(axis=1)
    )
    df["pareto_efficient_observation"] = _pareto_front(df, list(METRICS))
    df["diagnostic_status"] = np.where(
        df["balanced_objective_score"].notna(),
        "observed_outcome_comparison_only",
        "incomplete_outcomes_not_scored",
    )

    return df.drop(columns=["_diagnostic_condition"], errors="ignore")


def main() -> None:
    source = pd.read_csv(DATA_PATH)
    diagnostic = build_diagnostic(source)
    RESULTS.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {len(source)}; output rows: {len(diagnostic)}")
    print(f"Observed resource range: {source['resource_fraction'].min()} to {source['resource_fraction'].max()}")
    print(f"Rows flagged for verification: {int(diagnostic['anomaly_flag'].sum())}")
    print(f"Saved diagnostic: {OUTPUT_PATH}")
    print("No optimal allocation is claimed; allocation effects require comparable measurements at different allocations.")


if __name__ == "__main__":
    main()
