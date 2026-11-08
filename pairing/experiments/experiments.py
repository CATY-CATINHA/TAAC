"""Experimental pipeline for LocalADERH and HybridADERH.

- Controlled comparison: for each seed and dimensionality, both models receive the exact same training and test data.

- Relative Contrast: calculated once per configuration, as it describes the geometry of the data rather than the performance of a model.

- Intermediate results: the CSV file is updated after each run. If the process is interrupted, the configurations that have already been completed are not lost.

- Explicit parameters: these are recorded in the CSV file and in the metadata."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from aderh import ADERH
from pairing.local_pairing.local_aderh import LocalADERH
from pairing.hybrid_pairing.hybrid_aderh import HybridADERH
from pairing.lid_adaptive_pairing.lid_adaptive_aderh import LIDAdaptiveADERH

from .data_generation import DEFAULT_DIMENSIONS, make_nested_datasets
from .metrics import relative_contrast


# Experiment configuration
SMOKE_DIMENSIONS = (2, 50)
FULL_SEEDS = tuple(range(10))

MODEL_CONFIGS = {
    "original": {
        "class": ADERH,
        "parameters": {
            "n_estimators": 256,
            "n": 18,
            "contamination": 0.1,
        },
    },
    "local": {
        "class": LocalADERH,
        "parameters": {
            "n_estimators": 256,
            "n": 18,
            "contamination": 0.1,
            "k_neighbors": 5,
        },
    },
    "hybrid": {
        "class": HybridADERH,
        "parameters": {
            "n_estimators": 256,
            "n": 18,
            "contamination": 0.1,
            "k_neighbors": 5,
            "alpha": 0.5,
        },
    },
    "lid_adaptive_high_local": {
        "class": LIDAdaptiveADERH,
        "parameters": {
            "n_estimators": 256,
            "n": 18,
            "contamination": 0.1,
            "k_neighbors": 5,
            "alpha": 0.5,
            "lid_k": 20,
            "direction": "high_local",
        },
    },
    "lid_adaptive_high_random": {
        "class": LIDAdaptiveADERH,
        "parameters": {
            "n_estimators": 256,
            "n": 18,
            "contamination": 0.1,
            "k_neighbors": 5,
            "alpha": 0.5,
            "lid_k": 20,
            "direction": "high_random",
        },
    },
}

SUMMARY_METRICS = (
    "roc_auc",
    "average_precision",
    "relative_contrast",
    "test_score_std",
    "test_unique_score_count",
    "test_uncovered_fraction",
    "normal_uncovered_fraction",
    "anomaly_uncovered_fraction",
    "fit_time_seconds",
    "scoring_time_seconds",
)


def scale_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit MinMaxScaler on training data and transform both splits."""

    scaler = MinMaxScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_test),
        scaler,
    )


def split_indices(
    y: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified 70/30 train/test split."""

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )

    return next(splitter.split(np.zeros((y.shape[0], 1)), y))


def score_diagnostics(
    scores: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float | int]:
    """Calculate score distribution and hypersphere coverage diagnostics."""

    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true)

    if scores.ndim != 1 or y_true.ndim != 1 or scores.shape != y_true.shape:
        raise ValueError("scores and y_true must be one-dimensional and aligned")

    if not np.isin(y_true, (0, 1)).all() or np.unique(y_true).size != 2:
        raise ValueError("y_true must contain both binary classes 0 and 1")

    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    uncovered = scores == 1.0

    return {
        "test_score_mean": float(scores.mean()),
        "test_score_std": float(scores.std(ddof=0)),
        "test_score_min": float(scores.min()),
        "test_score_max": float(scores.max()),
        "test_unique_score_count": int(np.unique(scores).size),
        "test_uncovered_fraction": float(uncovered.mean()),
        "normal_score_mean": float(normal_scores.mean()),
        "anomaly_score_mean": float(anomaly_scores.mean()),
        "anomaly_normal_score_gap": float(
            anomaly_scores.mean() - normal_scores.mean()
        ),
        "normal_uncovered_fraction": float(uncovered[y_true == 0].mean()),
        "anomaly_uncovered_fraction": float(uncovered[y_true == 1].mean()),
    }


def aggregate_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and population standard deviation across seeds."""

    records = []

    for (model_name, dimensionality), group in raw_results.groupby(
        ["model", "dimensionality"], sort=True
    ):
        record = {
            "model": model_name,
            "dimensionality": int(dimensionality),
            "n_runs": int(group.shape[0]),
            "n_train": int(group["n_train"].iloc[0]),
            "n_test": int(group["n_test"].iloc[0]),
            "test_anomaly_prevalence_mean": float(
                group["test_anomaly_prevalence"].mean()
            ),
        }

        for metric in SUMMARY_METRICS:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=0))

        records.append(record)

    return pd.DataFrame(records)


def package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def create_figures(
    raw_results: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    """Generate comparison figures for all ADERH variants."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    plots = [
        (
            "roc_auc",
            "ROC-AUC",
            "ROC-AUC",
        ),
        (
            "average_precision",
            "Average Precision",
            "Average Precision",
        ),
        (
            "test_uncovered_fraction",
            "Uncovered fraction",
            "Uncovered fraction",
        ),
        (
            "fit_time_seconds",
            "Fit time (seconds)",
            "Fit time",
        ),
    ]

    for metric, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name in MODEL_CONFIGS:
            model_data = summary[summary["model"] == model_name]

            ax.errorbar(
                model_data["dimensionality"],
                model_data[f"{metric}_mean"],
                yerr=model_data[f"{metric}_std"],
                marker="o",
                capsize=4,
                label=model_name,
            )

        ax.set_xlabel("Dimensionality")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{prefix.capitalize()} - {title}")
        ax.set_xticks(sorted(raw_results["dimensionality"].unique()))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"{prefix}_{metric}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

    # Relative Contrast is a property of the dataset, not of the model.
    # Keep one observation per seed and dimensionality.
    contrast_data = (
        raw_results[
            ["dimensionality", "seed", "relative_contrast"]
        ]
        .drop_duplicates()
    )

    contrast_summary = (
        contrast_data.groupby("dimensionality")["relative_contrast"]
        .agg(["mean", "std"])
        .reset_index()
    )
    contrast_summary["std"] = contrast_summary["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        contrast_summary["dimensionality"],
        contrast_summary["mean"],
        yerr=contrast_summary["std"],
        marker="o",
        capsize=4,
    )
    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Relative Contrast")
    ax.set_title(f"{prefix.capitalize()} - Relative Contrast")
    ax.set_xticks(sorted(raw_results["dimensionality"].unique()))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / f"{prefix}_relative_contrast.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figure_paths.append(path)

    return figure_paths


def run_experiment(
    *,
    dimensions: Iterable[int],
    seeds: Iterable[int],
    prefix: str,
    results_dir: Path = Path("pairing/experiments/results"),
    figures_dir: Path = Path("pairing/experiments/figures"),
    reference_sample_size: int = 512,
    random_pair_count: int = 4096,
) -> dict[str, Path]:
    """Run all ADERH models on identical experimental configurations."""

    dimensions = tuple(int(value) for value in dimensions)
    seeds = tuple(int(value) for value in seeds)

    if not dimensions or not seeds:
        raise ValueError("dimensions and seeds must be non-empty")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / f"{prefix}_raw_results.csv"
    summary_path = results_dir / f"{prefix}_summary.csv"
    metadata_path = results_dir / f"{prefix}_metadata.json"

    rows: list[dict[str, object]] = []

    for seed in seeds:
        # Generate all dimensionalities from the same nested dataset.
        nested = make_nested_datasets(dimensions, random_state=seed)

        labels = nested[dimensions[0]].y
        train_indices, test_indices = split_indices(
            labels,
            random_state=seed,
        )

        for dimensionality in dimensions:
            dataset = nested[dimensionality]

            X_train = dataset.X[train_indices]
            X_test = dataset.X[test_indices]
            y_train = dataset.y[train_indices]
            y_test = dataset.y[test_indices]

            X_train_scaled, X_test_scaled, _ = scale_train_test(
                X_train,
                X_test,
            )

            # Relative Contrast is calculated once per dataset configuration.
            contrast = relative_contrast(
                X_train_scaled[y_train == 0],
                random_state=seed,
                reference_sample_size=reference_sample_size,
                random_pair_count=random_pair_count,
            )

            # Both models receive the same data and random seed.
            for model_name, config in MODEL_CONFIGS.items():
                model_class = config["class"]

                params = {
                    **config["parameters"],
                    "random_state": seed,
                }

                model = model_class(**params)

                fit_started = perf_counter()
                model.fit(X_train_scaled)
                fit_time = perf_counter() - fit_started

                scoring_started = perf_counter()
                scores = model.decision_function(X_test_scaled)
                scoring_time = perf_counter() - scoring_started

                diagnostics = score_diagnostics(scores, y_test)
                
                row = {
                    "model": model_name,
                    "dimensionality": dimensionality,
                    "seed": seed,
                    "roc_auc": float(roc_auc_score(y_test, scores)),
                    "average_precision": float(
                        average_precision_score(y_test, scores)
                    ),
                    "relative_contrast": contrast.value,
                    "rc_mean_random_distance": (
                        contrast.mean_random_distance
                    ),
                    "rc_mean_nearest_neighbor_distance": (
                        contrast.mean_nearest_neighbor_distance
                    ),
                    "rc_reference_sample_size": (
                        contrast.reference_sample_size
                    ),
                    "rc_random_pair_count": contrast.random_pair_count,
                    "fit_time_seconds": fit_time,
                    "scoring_time_seconds": scoring_time,
                    "n_train": int(train_indices.size),
                    "n_test": int(test_indices.size),
                    "train_anomaly_prevalence": float(y_train.mean()),
                    "test_anomaly_prevalence": float(y_test.mean()),
                    "model_n_estimators": int(params["n_estimators"]),
                    "model_n": int(params["n"]),
                    "model_contamination": float(params["contamination"]),
                    "model_random_state": int(params["random_state"]),
                    "model_hyperparameters": json.dumps(
                        params,
                        sort_keys=True,
                    ),
                    "preprocessing": (
                        "MinMaxScaler fit on training split only"
                    ),
                    "score_direction": "higher_is_more_anomalous",
                    **diagnostics,
                }

                rows.append(row)

                # Checkpoint after every completed model configuration.
                pd.DataFrame(rows).to_csv(raw_path, index=False)

                print(
                    f"completed model={model_name} "
                    f"dimension={dimensionality} seed={seed} "
                    f"roc_auc={row['roc_auc']:.4f} "
                    f"average_precision={row['average_precision']:.4f}",
                    flush=True,
                )

    raw_results = pd.DataFrame(rows)
    raw_results.to_csv(raw_path, index=False)

    summary = aggregate_results(raw_results)
    summary.to_csv(summary_path, index=False)

    figure_paths = create_figures(
        raw_results,
        summary,
        figures_dir,
        prefix,
    )

    metadata_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": prefix,
        "dimensions": list(dimensions),
        "seeds": list(seeds),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "scipy",
            )
        },
        "dataset": {
            "n_samples": 2000,
            "contamination": 0.05,
            "informative_dimensions": 2,
            "normal_distribution": "N([0, 0], I)",
            "anomaly_distribution": "N([4, 4], I)",
            "irrelevant_distribution": (
                "independent N(0, 1) for both classes"
            ),
        },
        "split": {
            "method": "StratifiedShuffleSplit",
            "train_fraction": 0.70,
            "test_fraction": 0.30,
            "same_indices_across_dimensions_within_seed": True,
        },
        "preprocessing": (
            "MinMaxScaler fitted on training data only"
        ),
        "relative_contrast": {
            "population": "normal training observations after scaling",
            "reference_sample_size_max": reference_sample_size,
            "random_pair_count": random_pair_count,
            "self_distances_excluded": True,
        },
        "models": {
            name: {
                "class": config["class"].__name__,
                "parameters": config["parameters"],
            }
            for name, config in MODEL_CONFIGS.items()
        },
        "score_direction": "higher decision_function values are more anomalous",
        "uncovered_score_semantics": (
            "score == 1.0 identifies observations not covered by any "
            "hypersphere under the pairing implementation's scoring semantics"
        ),
        "artifacts": {
            "raw_results": str(raw_path),
            "summary": str(summary_path),
            "figures": [str(path) for path in figure_paths],
        },
    }

    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "raw_results": raw_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "figures_directory": figures_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "mode",
        choices=("smoke", "full"),
        help="smoke: dimensions 2 and 50, one seed; full: all dimensions, 10 seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for smoke mode (default: 0)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("pairing/experiments/results"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("pairing/experiments/figures"),
    )
    parser.add_argument("--reference-sample-size", type=int, default=512)
    parser.add_argument("--random-pair-count", type=int, default=4096)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "smoke":
        dimensions = SMOKE_DIMENSIONS
        seeds = (args.seed,)
    else:
        dimensions = DEFAULT_DIMENSIONS
        seeds = FULL_SEEDS

    artifacts = run_experiment(
        dimensions=dimensions,
        seeds=seeds,
        prefix=args.mode,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        reference_sample_size=args.reference_sample_size,
        random_pair_count=args.random_pair_count,
    )

    print(
        json.dumps(
            {key: str(value) for key, value in artifacts.items()},
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())