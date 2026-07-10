from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pyplot(output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plotting_skipped.txt").write_text(
            "Plotting skipped because matplotlib is not importable in this environment.\n",
            encoding="utf-8",
        )
        return None
    return plt


def plot_predicted_vs_true(predictions: pd.DataFrame, target: str, output_path: Path) -> None:
    plt = _pyplot(output_path.parent)
    if plt is None or predictions.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for model, subset in predictions.groupby("model"):
        plt.scatter(subset[target], subset["prediction"], label=model, alpha=0.75)
    lo = min(predictions[target].min(), predictions["prediction"].min())
    hi = max(predictions[target].max(), predictions["prediction"].max())
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pareto(frame: pd.DataFrame, x: str, y: str, output_path: Path) -> None:
    plt = _pyplot(output_path.parent)
    if plt is None or frame.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(frame[x], frame[y], alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Selected multi-objective architectures")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
