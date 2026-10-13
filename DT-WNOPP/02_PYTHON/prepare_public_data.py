
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.home() / "Documents" / "DT-WNOPP"
DATA = ROOT / "03_DATA"

SOURCE = DATA / "ntnu_tput_all_Throughput.csv"
OUTPUT = DATA / "processed_measurements.csv"

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Download the public throughput CSV first: {SOURCE}"
    )

df = pd.read_csv(SOURCE, low_memory=False)
df.columns = df.columns.str.strip()

required = [
    "mbpsactual_downlink",
    "mbpsoffered_downlink",
    "meanjitterms_downlink",
    "meanloss_downlink",
    "bw",
    "scenario",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Unexpected source schema. Missing columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

numeric_cols = [
    "mbpsactual_downlink",
    "mbpsoffered_downlink",
    "meanjitterms_downlink",
    "meanloss_downlink",
    "bw",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols + ["scenario"]).copy()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=numeric_cols)

if len(df) < 20:
    raise ValueError(
        f"Only {len(df)} valid rows. Inspect the source file."
    )

max_offered = df["mbpsoffered_downlink"].max()
max_bandwidth = df["bw"].max()

if max_offered <= 0 or max_bandwidth <= 0:
    raise ValueError("Offered traffic and bandwidth must be positive.")

# Normalised inputs for the initial system-level twin.
df["traffic_load"] = (
    df["mbpsoffered_downlink"] / max_offered
).clip(0, 1)

df["resource_fraction"] = (
    df["bw"] / max_bandwidth
).clip(0.1, 1)

# Retain the source values and do not invent latency measurements.
result = pd.DataFrame({
    "time_step": np.arange(len(df)),
    "traffic_load": df["traffic_load"],
    "resource_fraction": df["resource_fraction"],
    "offered_downlink_source_units": df["mbpsoffered_downlink"],
    "throughput_observed_source_units": df["mbpsactual_downlink"],
    "jitter_observed_ms": df["meanjitterms_downlink"],
    "packet_loss_observed_source_units": df["meanloss_downlink"],
    "bandwidth_source_units": df["bw"],
    "scenario": df["scenario"].astype(str),
    "source": "Zenodo_5G_Campus_QoS",
})

result.to_csv(OUTPUT, index=False)

print(f"Source rows read: {len(pd.read_csv(SOURCE))}")
print(f"Valid observations: {len(result)}")
print(f"Saved processed data: {OUTPUT}")
print("\nAvailable columns:")
print(result.columns.tolist())
print("\nPreview:")
print(result.head())
