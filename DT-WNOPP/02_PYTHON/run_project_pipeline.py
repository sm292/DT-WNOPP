"""Run the DT-WNOPP MATLAB/Python workflow from one command.

Place this file in the DT-WNOPP project root, then run it with the project's
.venv Python. MATLAB is called with -batch, so matlab.engine is not required.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


def matlab_literal(path: Path) -> str:
    """Format a Windows path as a MATLAB character-vector literal."""
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def find_matlab(explicit_path: str | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"MATLAB executable not found: {candidate}")
        return candidate

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    installations = sorted(
        program_files.glob("MATLAB/R*/bin/matlab.exe"),
        key=lambda path: path.parts[-3],
        reverse=True,
    )
    if not installations:
        raise FileNotFoundError(
            "MATLAB was not found under Program Files. Pass --matlab-exe."
        )
    return installations[0]


def run_step(
    label: str,
    command: list[str],
    cwd: Path,
    dry_run: bool,
    display_command: str | None = None,
) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {label}", flush=True)
    shown = display_command or " ".join(
        f'"{part}"' if " " in part else part for part in command
    )
    print(shown, flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def evaluate_synthetic_latency(project_root: Path, python: Path, dry_run: bool) -> None:
    """Run the master controller's one-step latency model without MATLAB Engine."""
    inline = r'''
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

root = Path(sys.argv[1])
data_path = root / "03_DATA" / "dt_wnopp_simulation.csv"
results = root / "04_RESULTS"
df = pd.read_csv(data_path).sort_values("time_step").reset_index(drop=True)
features = ["traffic_load", "active_users", "resource_blocks", "throughput_mbps",
            "latency_ms", "packet_loss_pct", "energy_w"]
required = ["time_step", *features]
missing = [column for column in required if column not in df]
if missing:
    raise ValueError(f"Synthetic simulation data is missing columns: {missing}")
if len(df) < 30 or not np.isfinite(df[required].to_numpy(dtype=float)).all():
    raise ValueError("Synthetic data need at least 30 complete, finite rows.")
dataset = df[features].shift(1)
dataset["actual_latency_ms"] = df["latency_ms"]
dataset = dataset.dropna().reset_index(drop=True)
split = int(len(dataset) * 0.8)
if split < 2 or split >= len(dataset):
    raise ValueError("Not enough rows for a chronological train/test split.")
X_train, X_test = dataset[features].iloc[:split], dataset[features].iloc[split:]
y_train, y_test = dataset["actual_latency_ms"].iloc[:split], dataset["actual_latency_ms"].iloc[split:]
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3,
                              random_state=42, n_jobs=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
baseline = X_test["latency_ms"].to_numpy()
metrics = pd.DataFrame({
    "Persistence": [mean_absolute_error(y_test, baseline),
                    np.sqrt(mean_squared_error(y_test, baseline)), r2_score(y_test, baseline)],
    "Random Forest": [mean_absolute_error(y_test, pred),
                      np.sqrt(mean_squared_error(y_test, pred)), r2_score(y_test, pred)],
}, index=["MAE_ms", "RMSE_ms", "R2"]).T
comparison = pd.DataFrame({"actual_latency_ms": y_test.to_numpy(),
                           "persistence_prediction_ms": baseline,
                           "random_forest_prediction_ms": pred})
results.mkdir(parents=True, exist_ok=True)
comparison.to_csv(results / "latency_predictions.csv", index=False)
metrics.to_csv(results / "model_metrics.csv")
print("\nSynthetic latency holdout metrics")
print(metrics.round(4).to_string())
'''
    run_step(
        "Evaluate synthetic one-step latency (chronological 80/20 holdout)",
        [str(python), "-c", inline, str(project_root)],
        project_root,
        dry_run,
        display_command=f'"{python}" -c <latency evaluation> "{project_root}"',
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all DT-WNOPP simulation and public-data analysis stages."
    )
    parser.add_argument("--python-exe", help="Python executable; defaults to project .venv.")
    parser.add_argument("--matlab-exe", help="MATLAB executable; auto-detected by default.")
    parser.add_argument("--dry-run", action="store_true", help="Print workflow steps without running them.")
    args = parser.parse_args()

    root = (Path.home() / "Documents" / "DT-WNOPP").resolve()
    python = Path(args.python_exe).expanduser().resolve() if args.python_exe else root / ".venv" / "Scripts" / "python.exe"
    matlab = find_matlab(args.matlab_exe)
    required_paths = [
        root / "02_PYTHON" / "prepare_public_data.py",
        root / "02_PYTHON" / "calibrate_twin.py",
        root / "02_PYTHON" / "evaluate_throughput.py",
        root / "02_PYTHON" / "optimise_resources.py",
        root / "02_PYTHON" / "validate_and_screen.py",
        root / "01_MATLAB" / "dt_wnopp_simulation.m",
        root / "01_MATLAB" / "run_dt_wnopp_twin.m",
        root / "03_DATA" / "ntnu_tput_all_Throughput.csv",
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required project files not found:\n" + "\n".join(map(str, missing)))
    if not python.is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")

    matlab_dir = root / "01_MATLAB"
    data_dir = root / "03_DATA"
    results_dir = root / "04_RESULTS"
    simulation_csv = data_dir / "dt_wnopp_simulation.csv"
    processed_csv = data_dir / "processed_measurements.csv"
    twin_csv = results_dir / "twin_outputs.csv"

    print(f"Project: {root}")
    print(f"Python:  {python}")
    print(f"MATLAB:  {matlab}")
    print("This workflow regenerates derived CSV results; source CSVs are read only.")

    try:
        run_step("Prepare derived public measurements", [str(python), str(root / "02_PYTHON" / "prepare_public_data.py")], root, args.dry_run)

        simulation_call = (
            f"addpath({matlab_literal(matlab_dir)}); "
            f"dt_wnopp_simulation({matlab_literal(simulation_csv)});"
        )
        run_step("Generate deterministic synthetic MATLAB observations", [str(matlab), "-batch", simulation_call], root, args.dry_run)

        twin_call = (
            f"addpath({matlab_literal(matlab_dir)}); "
            f"run_dt_wnopp_twin({matlab_literal(processed_csv)}, {matlab_literal(twin_csv)});"
        )
        run_step("Run Simulink twin on processed public observations", [str(matlab), "-batch", twin_call], root, args.dry_run)

        run_step("Calibrate the twin on held-out configurations", [str(python), str(root / "02_PYTHON" / "calibrate_twin.py")], root, args.dry_run)
        run_step("Evaluate throughput models on held-out configurations", [str(python), str(root / "02_PYTHON" / "evaluate_throughput.py")], root, args.dry_run)
        run_step("Create resource optimisation diagnostic", [str(python), str(root / "02_PYTHON" / "optimise_resources.py")], root, args.dry_run)
        run_step("Run repeated grouped holdout metrics for all measured QoS targets", [str(python), str(root / "02_PYTHON" / "validate_and_screen.py")], root, args.dry_run)
        evaluate_synthetic_latency(root, python, args.dry_run)

    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline stopped: stage exited with status {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1

    if args.dry_run:
        print("\nDry run complete; no project files were changed.")
    else:
        print(f"\nPipeline completed. Review outputs in {results_dir}")
        print("Resource results are comparative diagnostics; no optimal allocation is inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
