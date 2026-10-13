"""Create separate, repeat-aware charts for throughput, jitter, and packet loss."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualization_common import apply_style, common_parser, output_dir, project_root, read_result, save_figure

TARGETS = {
    "throughput_observed_source_units": ("Throughput", "source units"),
    "jitter_observed_ms": ("Jitter", "ms"),
    "packet_loss_observed_source_units": ("Packet loss", "source units"),
}
MODEL_ORDER = ["Mean baseline", "Ridge", "Random Forest", "Extra Trees", "Gradient Boosting"]


def interpret_validation(metrics: pd.DataFrame) -> list[str]:
    lines = []
    for target, (label, unit) in TARGETS.items():
        part = metrics.loc[metrics.target == target]
        if part.empty:
            continue
        summary = part.groupby("model").agg(mae=("MAE", "mean"), rmse=("RMSE", "mean"), r2=("R2", "mean"))
        baseline = summary.loc["Mean baseline"] if "Mean baseline" in summary.index else None
        models = summary.drop(index="Mean baseline", errors="ignore")
        if models.empty:
            continue
        best = models.loc[models.mae.idxmin()]
        msg = f"{label}: lowest average MAE is {models.mae.idxmin()} at {best.mae:.4g} {unit}; average R² is {best.r2:.3f}."
        if best.r2 < 0:
            msg += " Negative average R² indicates weak generalisation on the held-out configurations."
        if baseline is not None:
            if best.mae < baseline.mae:
                msg += f" It improves MAE over the training-mean baseline ({baseline.mae:.4g}), but this alone does not make it reliable."
            else:
                msg += f" It does not improve MAE over the training-mean baseline ({baseline.mae:.4g})."
        lines.append(msg)
    lines.append("Error bars show split-to-split standard deviation across repeated grouped holdouts; they are not confidence intervals or independent experiment replications.")
    return lines


def generate_validation(root: Path, out: Path) -> tuple[list[Path], list[str]]:
    metrics = read_result(root, "validation_metrics_repeated.csv", ["target", "model", "MAE", "RMSE", "R2"])
    for col in ["MAE", "RMSE", "R2"]:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
    metrics = metrics.dropna(subset=["target", "model", "MAE", "RMSE", "R2"])
    apply_style()
    written = []
    for target, (label, unit) in TARGETS.items():
        part = metrics.loc[metrics.target == target]
        if part.empty:
            continue
        order = [name for name in MODEL_ORDER if name in set(part.model)]
        summary = part.groupby("model").agg({c: ["mean", "std"] for c in ["MAE", "RMSE", "R2"]})
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
        fig.suptitle(f"{label} prediction across grouped holdout splits", fontsize=15, fontweight="bold")
        for ax, metric, title, ylabel in zip(
            axes, ["MAE", "RMSE", "R2"], ["Mean absolute error", "Root mean squared error", "R²"],
            [f"{label} error ({unit})", f"{label} error ({unit})", "R² (unitless)"]
        ):
            means = [summary.loc[name, (metric, "mean")] for name in order]
            spread = [summary.loc[name, (metric, "std")] for name in order]
            x = np.arange(len(order))
            colors = ["#8795a5" if name == "Mean baseline" else "#277da1" for name in order]
            ax.bar(x, means, yerr=spread, capsize=3, color=colors, alpha=0.88, edgecolor="none")
            if metric == "R2":
                ax.axhline(0, color="#b0443c", linewidth=1.2, linestyle="--")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x, order, rotation=28, ha="right", rotation_mode="anchor")
            ax.grid(axis="x", visible=False)
        fig.subplots_adjust(left=0.055, right=0.99, top=0.84, bottom=0.31, wspace=0.36)
        fig.text(0.5, 0.035, "Means across repeated grouped holdouts; whiskers show ±1 split-to-split SD.", ha="center", color="#526273", fontsize=9)
        path = out / f"validation_{target.replace('_observed_source_units','').replace('_observed_ms','')}.png"
        save_figure(fig, path)
        written.append(path)
    return written, interpret_validation(metrics)


def main() -> None:
    parser = common_parser(__doc__ or "Plot validation outputs")
    args = parser.parse_args()
    root = project_root(__file__, args.project_root)
    out = output_dir(root, args.output_dir)
    _, notes = generate_validation(root, out)
    (out / "validation_interpretation.txt").write_text("\n\n".join(notes) + "\n", encoding="utf-8")
    print(f"Validation plots and interpretation saved in {out}")

if __name__ == "__main__":
    main()
