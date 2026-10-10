#import sys
#print(sys.executable)



from pathlib import Path
from datetime import datetime
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matlab.engine

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# DT-WNOPP MASTER CONTROLLER
# MATLAB simulation -> Python prediction -> evaluation
# ==========================================================

PROJECT_ROOT = Path.home() / "Documents" / "DT-WNOPP"
MATLAB_DIR = PROJECT_ROOT / "01_MATLAB"
PYTHON_DIR = PROJECT_ROOT / "02_PYTHON"
DATA_DIR = PROJECT_ROOT / "03_DATA"
RESULTS_DIR = PROJECT_ROOT / "04_RESULTS"

INPUT_FILE = DATA_DIR / "dt_wnopp_simulation.csv"


def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_pipeline():
    eng = None

    # Create output folders if needed
    for folder in [
        MATLAB_DIR, PYTHON_DIR, DATA_DIR, RESULTS_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    try:
        # --------------------------------------------------
        # STAGE 1: START MATLAB
        # --------------------------------------------------
        log("Starting MATLAB Engine...")

        eng = matlab.engine.start_matlab()

        log("MATLAB started successfully.")

        # Add folder containing the MATLAB function
        eng.addpath(str(MATLAB_DIR), nargout=0)

        # --------------------------------------------------
        # STAGE 2: RUN MATLAB SIMULATION
        # --------------------------------------------------
        log("Running DT-WNOPP MATLAB simulation...")

        eng.dt_wnopp_simulation(
            str(INPUT_FILE),
            nargout=0
        )

        if not INPUT_FILE.exists():
            raise FileNotFoundError(
                f"MATLAB did not create {INPUT_FILE}"
            )

        log("Simulation completed; CSV file found.")

        # --------------------------------------------------
        # STAGE 3: LOAD AND VALIDATE DATA
        # --------------------------------------------------
        log("Loading simulation data into Python...")

        df = pd.read_csv(INPUT_FILE)

        required = [
            "time_step",
            "traffic_load",
            "active_users",
            "resource_blocks",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_pct",
            "energy_w",
        ]

        missing = [
            col for col in required if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        df = df.sort_values("time_step").reset_index(drop=True)

        if len(df) < 20:
            raise ValueError(
                "Insufficient rows for time-series evaluation."
            )

        if not np.isfinite(
            df[required].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Input data contain missing or infinite values."
            )

        log(f"Loaded {len(df)} simulation observations.")

        # --------------------------------------------------
        # STAGE 4: PREPARE NEXT-STEP PREDICTION
        # --------------------------------------------------
        features = [
            "traffic_load",
            "active_users",
            "resource_blocks",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_pct",
            "energy_w",
        ]

        dataset = df[features].shift(1)
        dataset["target_latency_ms"] = df["latency_ms"]

        dataset = dataset.dropna().reset_index(drop=True)

        X = dataset[features]
        y = dataset["target_latency_ms"]

        split = int(len(dataset) * 0.80)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        if len(X_test) == 0:
            raise ValueError("The test set is empty.")

        log(
            f"Training observations: {len(X_train)}; "
            f"test observations: {len(X_test)}."
        )

        # --------------------------------------------------
        # STAGE 5: BASELINE AND RANDOM FOREST
        # --------------------------------------------------
        log("Training Random Forest model...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Baseline: previous observed latency
        baseline = X_test["latency_ms"].to_numpy()

        def metrics(actual, predicted):
            return {
                "MAE_ms": mean_absolute_error(
                    actual, predicted
                ),
                "RMSE_ms": np.sqrt(
                    mean_squared_error(actual, predicted)
                ),
                "R2": r2_score(actual, predicted),
            }

        results = pd.DataFrame({
            "Persistence": metrics(y_test, baseline),
            "Random Forest": metrics(y_test, predictions),
        }).T

        log("Model evaluation completed.")

        print("\nMODEL PERFORMANCE")
        print(results.round(4).to_string())

        # --------------------------------------------------
        # STAGE 6: SAVE RESULTS
        # --------------------------------------------------
        comparison = pd.DataFrame({
            "actual_latency_ms": y_test.to_numpy(),
            "persistence_prediction_ms": baseline,
            "random_forest_prediction_ms": predictions,
        })

        comparison.to_csv(
            RESULTS_DIR / "latency_predictions.csv",
            index=False,
        )

        results.to_csv(
            RESULTS_DIR / "model_metrics.csv"
        )

        # --------------------------------------------------
        # STAGE 7: GENERATE GRAPH
        # --------------------------------------------------
        log("Generating prediction graph...")

        fig, ax = plt.subplots(figsize=(11, 5))

        n_plot = min(150, len(y_test))

        ax.plot(
            y_test.to_numpy()[:n_plot],
            label="Actual latency",
        )

        ax.plot(
            predictions[:n_plot],
            label="Random Forest prediction",
        )

        ax.set_xlabel("Test sample")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(
            "DT-WNOPP One-Step-Ahead Latency Prediction"
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            RESULTS_DIR / "latency_prediction.png",
            dpi=300,
        )

        plt.close(fig)

        # Save a short run summary
        summary_file = RESULTS_DIR / "run_summary.txt"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("DT-WNOPP prototype run summary\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"Observations: {len(df)}\n\n")
            f.write(results.round(4).to_string())
            f.write("\n\nNote: results use synthetic simulation data.\n")

        log("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY.")
        log(f"Results folder: {RESULTS_DIR}")

    except Exception:
        log("PIPELINE FAILED.")
        traceback.print_exc()
        raise

    finally:
        if eng is not None:
            log("Closing MATLAB Engine...")
            eng.quit()
            log("MATLAB Engine closed.")


if __name__ == "__main__":
    run_pipeline()
