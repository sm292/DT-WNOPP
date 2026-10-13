"""Visualize measured resource scenarios and matched Simulink proxy results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualization_common import apply_style, common_parser, output_dir, project_root, read_result, save_figure


def scenario_interpretation(scenarios: pd.DataFrame) -> tuple[list[str], int]:
    groups_per_fraction = scenarios.groupby("resource_fraction").config_group.nunique()
    group_levels = scenarios.groupby("config_group").resource_fraction.nunique()
    overlap = int(group_levels.gt(1).sum())
    lines = [f"The scenario summary contains {scenarios.config_group.nunique()} configuration groups across resource fractions {', '.join(f'{x:g}' for x in sorted(scenarios.resource_fraction.unique()))}; {overlap} groups are represented at more than one fraction."]
    for fraction, part in scenarios.groupby("resource_fraction", sort=True):
        n = pd.to_numeric(part["n"], errors="coerce").fillna(0)
        def weighted(col: str) -> float:
            v = pd.to_numeric(part[col], errors="coerce")
            return float((v * n).sum() / n[v.notna()].sum())
        lines.append(f"At fraction {fraction:g}: {int(n.sum())} observations; weighted mean throughput {weighted('throughput'):.4g}, jitter {weighted('jitter'):.4g} ms, and packet loss {weighted('loss'):.4g} source units.")
    if overlap == 0:
        lines.append("No configuration is observed at both resource fractions. These measured averages are confounded comparisons, not allocation effects.")
    return lines, overlap


def proxy_interpretation(deltas: pd.DataFrame) -> list[str]:
    changed = deltas.loc[deltas.candidate_fraction != deltas.baseline_fraction]
    lines = []
    for _, row in changed.iterrows():
        direction = "decreased" if row.mean_absolute_change < 0 else "increased" if row.mean_absolute_change > 0 else "did not change"
        lines.append(f"From resource fraction {row.baseline_fraction:g} to {row.candidate_fraction:g}, the {row.metric.replace('_',' ')} {direction} by {abs(row.mean_percent_change):.2f}% ({row.baseline_mean:.4g} to {row.candidate_mean:.4g}; {int(row.matched_samples)} matched samples).")
    lines.append("These are Simulink proxy outputs. The model's equations directly make throughput increase with resource fraction and latency decrease with it; packet loss depends on traffic load, not the resource fraction. This comparison does not establish real network gains.")
    return lines


def generate_system(root: Path, out: Path) -> tuple[list[Path], list[str], list[str]]:
    scenarios = read_result(root, "matched_resource_scenario_summary.csv", ["config_group", "resource_fraction", "n", "throughput", "jitter", "loss"])
    deltas = read_result(root, "candidate_simulations/matched_proxy_deltas.csv", ["baseline_fraction", "candidate_fraction", "metric", "matched_samples", "baseline_mean", "candidate_mean", "mean_percent_change"])
    for col in ["resource_fraction", "n", "throughput", "jitter", "loss"]:
        scenarios[col] = pd.to_numeric(scenarios[col], errors="coerce")
    for col in ["baseline_fraction", "candidate_fraction", "matched_samples", "baseline_mean", "candidate_mean", "mean_percent_change"]:
        deltas[col] = pd.to_numeric(deltas[col], errors="coerce")
    scenarios = scenarios.dropna(subset=["resource_fraction", "n"])
    deltas = deltas.dropna(subset=["baseline_fraction", "candidate_fraction", "metric"])
    apply_style()
    written: list[Path] = []

    # Each dot is a configuration-level measured average; a diamond is the observation-weighted mean.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    fig.suptitle("Measured scenario outcomes by observed resource fraction", fontsize=15, fontweight="bold")
    fractions = sorted(scenarios.resource_fraction.unique())
    series = [("throughput", "Throughput (source units)"), ("jitter", "Jitter (ms)"), ("loss", "Packet loss (source units)")]
    palette = ["#277da1", "#f28e2b", "#55a868", "#b279a2"]
    for ax, (col, ylabel) in zip(axes, series):
        for i, fraction in enumerate(fractions):
            part = scenarios.loc[scenarios.resource_fraction == fraction]
            jitter = np.linspace(-0.065, 0.065, len(part)) if len(part) > 1 else np.array([0.0])
            vals = part[col].to_numpy(dtype=float)
            ax.scatter(np.full(len(part), i) + jitter, vals, s=34, alpha=0.72, color=palette[i % len(palette)], label=f"{fraction:g} per configuration")
            counts = part["n"].to_numpy(dtype=float)
            valid = np.isfinite(vals) & np.isfinite(counts) & (counts > 0)
            if valid.any():
                mean = float(np.average(vals[valid], weights=counts[valid]))
                ax.scatter(i, mean, marker="D", s=95, color="#202d3b", edgecolor="white", linewidth=0.8, zorder=4, label="Observation-weighted mean" if i == 0 else None)
        ax.set_xticks(range(len(fractions)), [f"{f:g}" for f in fractions])
        ax.set_xlabel("Resource fraction")
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, -0.09, "Dots are different configuration groups. No group appears at both fractions; do not read this as a matched resource effect.", ha="center", color="#526273", fontsize=9)
    scenario_path = out / "measured_resource_scenarios.png"
    save_figure(fig, scenario_path)
    written.append(scenario_path)

    # Paired proxy baseline/candidate bars, with a separate axis per proxy measure.
    changed = deltas.loc[deltas.candidate_fraction != deltas.baseline_fraction]
    metric_order = ["latency_proxy_ms", "throughput_proxy_normalised", "packet_loss_proxy_pct"]
    titles = ["Latency proxy (ms)", "Normalised throughput proxy", "Packet-loss proxy (model units)"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.0))
    fig.suptitle("Matched Simulink proxy comparison", fontsize=15, fontweight="bold")
    for ax, metric, title in zip(axes, metric_order, titles):
        part = changed.loc[changed.metric == metric].sort_values("candidate_fraction")
        if part.empty:
            ax.set_visible(False)
            continue
        labels = [f"{row.baseline_fraction:g} → {row.candidate_fraction:g}" for row in part.itertuples()]
        x = np.arange(len(part))
        width = 0.36
        ax.bar(x - width/2, part.baseline_mean, width, label="Baseline", color="#8795a5")
        ax.bar(x + width/2, part.candidate_mean, width, label="Candidate", color="#277da1")
        for j, row in enumerate(part.itertuples()):
            ax.text(j, 0.94, f"{row.mean_percent_change:+.1f}% · n={int(row.matched_samples)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Baseline → candidate fraction")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, max(float(part.baseline_mean.max()), float(part.candidate_mean.max())) * 1.28)
        ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.2, wspace=0.4)
    fig.legend(handles=[Patch(facecolor="#8795a5", label="Baseline"), Patch(facecolor="#277da1", label="Candidate")], loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False)
    fig.text(0.5, 0.035, "Current Simulink proxy outputs; not independently measured network outcomes.", ha="center", color="#526273", fontsize=9)
    proxy_path = out / "matched_simulink_proxy_results.png"
    save_figure(fig, proxy_path)
    written.append(proxy_path)

    scenario_notes, _ = scenario_interpretation(scenarios)
    proxy_notes = proxy_interpretation(deltas)
    (out / "system_efficiency_interpretation.txt").write_text("MEASURED SCENARIOS\n\n" + "\n\n".join(scenario_notes) + "\n\nSIMULINK PROXIES\n\n" + "\n\n".join(proxy_notes) + "\n", encoding="utf-8")
    return written, scenario_notes, proxy_notes


def main() -> None:
    parser = common_parser(__doc__ or "Plot system results")
    args = parser.parse_args()
    root = project_root(__file__, args.project_root)
    out = output_dir(root, args.output_dir)
    _, scenario_notes, proxy_notes = generate_system(root, out)
    print(f"System figures and interpretation saved in {out}")

if __name__ == "__main__":
    main()
