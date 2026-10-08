"""CLI and experiment pipeline for the original ADERH baseline."""

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

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from .data_generation import DEFAULT_DIMENSIONS, make_nested_datasets
from .metrics import relative_contrast
from .visualization import create_all_figures

SMOKE_DIMENSIONS = (2, 50)
FULL_SEEDS = tuple(range(10))
SUMMARY_METRICS = (
    "roc_auc",
    "average_precision",
    "relative_contrast",
    "fit_time_seconds",
    "scoring_time_seconds",
)


def _load_aderh_class():
    try:
        from aderh import ADERH
    except ImportError as exc:  # pragma: no cover - exercised in missing-dep envs
        raise RuntimeError(
            'The official ADERH package is unavailable. Install with '
            '`python -m pip install -e ".[dev]"`.'
        ) from exc
    return ADERH


def scale_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit MinMax preprocessing on training data and transform both splits."""

    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def _split_indices(y: np.ndarray, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state,
    )
    return next(splitter.split(np.zeros((y.shape[0], 1)), y))


def _write_csv(rows: list[dict[str, object]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def aggregate_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate population mean/std across seeds for each dimensionality."""

    records: list[dict[str, float | int]] = []
    for dimensionality, group in raw_results.groupby("dimensionality", sort=True):
        record: dict[str, float | int] = {
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


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def run_experiment(
    *,
    dimensions: Iterable[int],
    seeds: Iterable[int],
    prefix: str,
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
    reference_sample_size: int = 512,
    random_pair_count: int = 4_096,
) -> dict[str, Path]:
    """Run configurations, checkpoint raw CSV rows, aggregate, and plot."""

    dimensions = tuple(int(value) for value in dimensions)
    seeds = tuple(int(value) for value in seeds)
    if not dimensions or not seeds:
        raise ValueError("dimensions and seeds must be non-empty")

    ADERH = _load_aderh_class()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / f"{prefix}_raw_results.csv"
    summary_path = results_dir / f"{prefix}_summary.csv"
    metadata_path = results_dir / f"{prefix}_metadata.json"

    rows: list[dict[str, object]] = []
    for seed in seeds:
        nested = make_nested_datasets(dimensions, random_state=seed)
        labels = nested[dimensions[0]].y
        train_indices, test_indices = _split_indices(labels, random_state=seed)

        for dimensionality in dimensions:
            dataset = nested[dimensionality]
            X_train = dataset.X[train_indices]
            X_test = dataset.X[test_indices]
            y_train = dataset.y[train_indices]
            y_test = dataset.y[test_indices]
            X_train_scaled, X_test_scaled, _ = scale_train_test(X_train, X_test)

            # Ground-truth labels are used here only for a separate geometric
            # diagnostic, never for preprocessing, model fitting, or tuning.
            contrast = relative_contrast(
                X_train_scaled[y_train == 0],
                random_state=seed,
                reference_sample_size=reference_sample_size,
                random_pair_count=random_pair_count,
            )

            # Only random_state is supplied: all algorithm hyperparameters are
            # the official ADERH 1.0.0 documented defaults.
            model = ADERH(random_state=seed)
            fit_started = perf_counter()
            model.fit(X_train_scaled)
            fit_time = perf_counter() - fit_started

            scoring_started = perf_counter()
            scores = model.decision_function(X_test_scaled)
            scoring_time = perf_counter() - scoring_started
            params = model.get_params(deep=False)

            rows.append(
                {
                    "dimensionality": dimensionality,
                    "seed": seed,
                    "roc_auc": float(roc_auc_score(y_test, scores)),
                    "average_precision": float(
                        average_precision_score(y_test, scores)
                    ),
                    "relative_contrast": contrast.value,
                    "rc_mean_random_distance": contrast.mean_random_distance,
                    "rc_mean_nearest_neighbor_distance": (
                        contrast.mean_nearest_neighbor_distance
                    ),
                    "rc_reference_sample_size": contrast.reference_sample_size,
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
                    "model_hyperparameters": json.dumps(params, sort_keys=True),
                    "preprocessing": "MinMaxScaler fit on training split only",
                    "score_direction": "higher_is_more_anomalous",
                }
            )
            # Preserve completed configurations if a later one is interrupted.
            _write_csv(rows, raw_path)
            print(
                f"completed dimension={dimensionality} seed={seed} "
                f"roc_auc={rows[-1]['roc_auc']:.4f} "
                f"average_precision={rows[-1]['average_precision']:.4f}",
                flush=True,
            )

    raw_results = _write_csv(rows, raw_path)
    summary = aggregate_results(raw_results)
    summary.to_csv(summary_path, index=False)
    figure_paths = create_all_figures(
        raw_results,
        summary,
        output_dir=figures_dir,
        prefix=prefix,
    )

    metadata_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": prefix,
        "dimensions": list(dimensions),
        "seeds": list(seeds),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("aderh", "numpy", "pandas", "scikit-learn", "matplotlib")
        },
        "dataset": {
            "n_samples": 2_000,
            "contamination": 0.05,
            "informative_dimensions": 2,
            "normal_distribution": "N([0, 0], I)",
            "anomaly_distribution": "N([4, 4], I)",
            "irrelevant_distribution": "independent N(0, 1) for both classes",
        },
        "split": {
            "method": "StratifiedShuffleSplit",
            "train_fraction": 0.70,
            "test_fraction": 0.30,
            "same_indices_across_dimensions_within_seed": True,
        },
        "preprocessing": "MinMaxScaler fitted on training data only",
        "relative_contrast": {
            "population": "normal training observations after scaling",
            "reference_sample_size_max": reference_sample_size,
            "random_pair_count": random_pair_count,
            "self_distances_excluded": True,
        },
        "aderh": {
            "implementation": "official PyPI package",
            "score_direction": "higher decision_function values are more anomalous",
            "hyperparameters": json.loads(rows[0]["model_hyperparameters"]),
        },
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("smoke", "full"),
        help="smoke runs dimensions 2 and 50 with one seed; full runs all dimensions and 10 seeds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for smoke mode (default: 0)",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--reference-sample-size", type=int, default=512)
    parser.add_argument("--random-pair-count", type=int, default=4_096)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
