"""Shared paths, inputs, and styling for DT-WNOPP result visualizations."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_PROJECT = Path(r"C:\Users\HP\Documents\DT-WNOPP")


def project_root(script_file: str, requested: str | None = None) -> Path:
    if requested:
        root = Path(requested).expanduser().resolve()
    elif Path(script_file).resolve().parent.name.lower() == "02_python":
        root = Path(script_file).resolve().parent.parent
    else:
        root = DEFAULT_PROJECT
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder not found: {root}. Pass --project-root.")
    return root


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--project-root", help="DT-WNOPP project folder (defaults to the known project path).")
    parser.add_argument("--output-dir", help="Folder for generated figures and interpretations.")
    return parser


def output_dir(root: Path, requested: str | None) -> Path:
    target = Path(requested).expanduser().resolve() if requested else root / "04_RESULTS" / "visualizations"
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_result(root: Path, relative: str, required: list[str]) -> pd.DataFrame:
    path = root / "04_RESULTS" / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    frame = pd.read_csv(path)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}; found {frame.columns.tolist()}")
    return frame


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.titleweight": "bold", "axes.labelcolor": "#263445",
        "xtick.color": "#35475b", "ytick.color": "#35475b",
        "text.color": "#202d3b", "grid.color": "#d8e0e8",
        "grid.alpha": 0.65, "font.size": 10,
        "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
