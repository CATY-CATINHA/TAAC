"""Publication-oriented plots for ADERH dimensionality experiments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Matplotlib otherwise tries to lock a user-home cache, which may be read-only
# in containers and managed research runners.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aderh-distance-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter


def _finish_axis(ax: plt.Axes, dimensions: list[int], ylabel: str) -> None:
    ax.set_xlabel("Total dimensionality")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(dimensions))
    ax.xaxis.set_major_formatter(FixedFormatter([str(value) for value in dimensions]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, output_stem: Path) -> list[Path]:
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.tight_layout()
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _single_metric_plot(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output_stem: Path,
    baseline: float | None = None,
    baseline_label: str | None = None,
) -> list[Path]:
    dimensions = summary["dimensionality"].astype(int).tolist()
    means = summary[f"{metric}_mean"].to_numpy()
    stds = summary[f"{metric}_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(
        dimensions,
        means,
        yerr=stds,
        marker="o",
        markersize=5,
        linewidth=1.8,
        capsize=3,
        color="#1f5a94",
        label="ADERH" if baseline is not None else None,
    )
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#9b2c2c",
            linestyle="--",
            linewidth=1.3,
            label=baseline_label,
        )
        ax.legend(frameon=False)
    _finish_axis(ax, dimensions, ylabel)
    return _save(fig, output_stem)


def create_all_figures(
    raw_results: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    """Create PNG and vector-PDF versions of all requested figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prevalence = float(raw_results["test_anomaly_prevalence"].mean())
    paths: list[Path] = []
    paths.extend(
        _single_metric_plot(
            summary,
            metric="relative_contrast",
            ylabel="Relative contrast",
            output_stem=output_dir / f"{prefix}_relative_contrast",
        )
    )
    paths.extend(
        _single_metric_plot(
            summary,
            metric="roc_auc",
            ylabel="ROC-AUC",
            output_stem=output_dir / f"{prefix}_roc_auc",
            baseline=0.5,
            baseline_label="Random ranking (0.5)",
        )
    )
    paths.extend(
        _single_metric_plot(
            summary,
            metric="average_precision",
            ylabel="Average precision",
            output_stem=output_dir / f"{prefix}_average_precision",
            baseline=prevalence,
            baseline_label=f"Test prevalence ({prevalence:.3f})",
        )
    )

    dimensions = summary["dimensionality"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for metric, label, color in (
        ("test_uncovered_fraction", "All test observations", "#4b5563"),
        ("normal_uncovered_fraction", "Normal", "#1f5a94"),
        ("anomaly_uncovered_fraction", "Anomaly", "#d97706"),
    ):
        ax.errorbar(
            dimensions,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            capsize=3,
            label=label,
            color=color,
        )
    _finish_axis(ax, dimensions, "Fraction with score = 1 (uncovered)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    paths.extend(_save(fig, output_dir / f"{prefix}_uncovered_fraction"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for metric, label, color in (
        ("fit_time_seconds", "Fit", "#1f5a94"),
        ("scoring_time_seconds", "Score", "#d97706"),
    ):
        ax.errorbar(
            dimensions,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            capsize=3,
            label=label,
            color=color,
        )
    _finish_axis(ax, dimensions, "Wall-clock time (seconds)")
    ax.legend(frameon=False)
    paths.extend(_save(fig, output_dir / f"{prefix}_runtime"))
    return paths
