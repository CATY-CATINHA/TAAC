"""

MLE-LID analysis on the controlled ADERH dimensionality baseline.

This script deliberately reuses the original ADERH baseline:

- synthetic data generator;
- dimensionalities;
- random seeds;
- stratified train/test split;
- MinMax preprocessing.

This guarantees that the LID analysis is performed on exactly the same
experimental configurations used in the original ADERH baseline.

For each dimensionality and random seed, the script:

1. regenerates the same nested synthetic dataset;
2. reproduces the same stratified 70/30 split;
3. applies the same train-fitted MinMaxScaler;
4. estimates MLE-LID for every training observation;
5. stores point-level LID values;
6. computes descriptive statistics for each configuration;
7. aggregates results across seeds;
8. stores metadata describing the experiment.

Important:
LID estimation itself is fully unsupervised.
Ground-truth labels are used only to reproduce the same stratified
train/test split as the original baseline.

"""

from __future__ import annotations

import argparse
import json
import platform
import sys

from importlib import metadata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from aderh_distance.data_generation import (
    DEFAULT_DIMENSIONS,
    make_nested_datasets,
)

from aderh_distance.experiments import (
    FULL_SEEDS,
    SMOKE_DIMENSIONS,
    _split_indices,
    scale_train_test,
)

from lid_analysis.lid import estimate_lid


# =========================================================
# Configuration
# =========================================================

DEFAULT_K = 20


# Statistics that will later be aggregated across seeds
LID_SUMMARY_METRICS = (
    "lid_mean",
    "lid_median",
    "lid_std",
    "lid_min",
    "lid_q25",
    "lid_q75",
    "lid_max",
)


# =========================================================
# Helper functions
# =========================================================

def summarize_lid(
    lid_values: np.ndarray,
) -> dict[str, float]:
    """
    Calculate descriptive statistics for one LID distribution.

    Parameters
    ----------
    lid_values : numpy.ndarray
        LID estimates for all training observations in one
        dimensionality/seed configuration.

    Returns
    -------
    dict
        Summary statistics describing the LID distribution.
    """

    lid_values = np.asarray(
        lid_values,
        dtype=float,
    )

    if lid_values.ndim != 1:
        raise ValueError(
            "lid_values must be one-dimensional."
        )

    if lid_values.size == 0:
        raise ValueError(
            "lid_values cannot be empty."
        )

    if not np.all(np.isfinite(lid_values)):
        raise ValueError(
            "lid_values contains NaN or infinite values."
        )

    if np.any(lid_values <= 0):
        raise ValueError(
            "LID estimates must be positive."
        )

    return {
        "lid_mean": float(
            np.mean(lid_values)
        ),
        "lid_median": float(
            np.median(lid_values)
        ),
        "lid_std": float(
            np.std(lid_values, ddof=0)
        ),
        "lid_min": float(
            np.min(lid_values)
        ),
        "lid_q25": float(
            np.percentile(lid_values, 25)
        ),
        "lid_q75": float(
            np.percentile(lid_values, 75)
        ),
        "lid_max": float(
            np.max(lid_values)
        ),
    }


def aggregate_by_dimension(
    by_seed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate LID statistics across seeds for each dimensionality.

    For each metric, the function calculates:

    - mean across seeds;
    - population standard deviation across seeds.

    This mirrors the aggregation philosophy of the ADERH baseline.
    """

    required_columns = {
        "dimensionality",
        "seed",
        "k",
        "n_train",
        *LID_SUMMARY_METRICS,
    }

    missing = required_columns.difference(
        by_seed.columns
    )

    if missing:
        raise ValueError(
            "Missing columns in LID results: "
            f"{sorted(missing)}"
        )

    records = []

    for dimensionality, group in by_seed.groupby(
        "dimensionality",
        sort=True,
    ):

        record = {
            "dimensionality": int(
                dimensionality
            ),
            "n_runs": int(
                group.shape[0]
            ),
            "n_train": int(
                group["n_train"].iloc[0]
            ),
            "k": int(
                group["k"].iloc[0]
            ),
        }

        for metric in LID_SUMMARY_METRICS:

            record[
                f"{metric}_mean"
            ] = float(
                group[metric].mean()
            )

            record[
                f"{metric}_std"
            ] = float(
                group[metric].std(
                    ddof=0
                )
            )

        records.append(record)

    return pd.DataFrame(records)


def _package_version(
    distribution: str,
) -> str:
    """
    Return installed package version for experiment metadata.
    """

    try:
        return metadata.version(
            distribution
        )

    except metadata.PackageNotFoundError:
        return "not-installed"


def _write_checkpoint(
    summary_rows: list[dict[str, object]],
    path: Path,
) -> None:
    """
    Save completed dimension/seed configurations during execution.

    This avoids losing all completed work if a full run is interrupted.
    """

    if not summary_rows:
        return

    frame = pd.DataFrame(
        summary_rows
    )

    frame.to_csv(
        path,
        index=False,
    )


# =========================================================
# Main LID experiment
# =========================================================

def run_lid_analysis(
    *,
    dimensions: Iterable[int],
    seeds: Iterable[int],
    k: int = DEFAULT_K,
    prefix: str,
    results_dir: Path = Path(
        "lid_analysis/results"
    ),
) -> dict[str, Path]:
    """
    Run MLE-LID analysis on exactly the same synthetic configurations
    used by the controlled ADERH baseline.

    Parameters
    ----------
    dimensions : iterable of int
        Dimensionalities to analyse.

    seeds : iterable of int
        Random seeds.

    k : int
        Number of nearest neighbours used by MLE-LID.

    prefix : str
        Name used for output files, e.g. "smoke" or "full".

    results_dir : pathlib.Path
        Output directory.

    Returns
    -------
    dict
        Paths of generated result files.
    """

    dimensions = tuple(
        int(value)
        for value in dimensions
    )

    seeds = tuple(
        int(value)
        for value in seeds
    )

    if not dimensions:
        raise ValueError(
            "dimensions cannot be empty."
        )

    if not seeds:
        raise ValueError(
            "seeds cannot be empty."
        )

    if k <= 1:
        raise ValueError(
            "k must be greater than 1."
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------

    points_path = (
        results_dir
        / f"{prefix}_lid_points.csv"
    )

    by_seed_path = (
        results_dir
        / f"{prefix}_lid_by_seed.csv"
    )

    summary_path = (
        results_dir
        / f"{prefix}_lid_summary.csv"
    )

    metadata_path = (
        results_dir
        / f"{prefix}_lid_metadata.json"
    )

    # -----------------------------------------------------
    # Collected results
    # -----------------------------------------------------

    point_rows: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    # =====================================================
    # Experimental loop
    # =====================================================

    for seed in seeds:

        # -------------------------------------------------
        # Generate EXACTLY the same nested datasets
        # used in the original baseline.
        # -------------------------------------------------

        nested = make_nested_datasets(
            dimensions=dimensions,
            random_state=seed,
        )

        # Labels are identical across dimensionalities.
        # They are used only to reproduce the stratified split.
        labels = nested[
            dimensions[0]
        ].y

        # -------------------------------------------------
        # Same train/test split used by ADERH baseline
        # -------------------------------------------------

        train_indices, test_indices = _split_indices(
            labels,
            random_state=seed,
        )

        # -------------------------------------------------
        # Same indices reused across every dimensionality
        # inside this seed.
        # -------------------------------------------------

        for dimensionality in dimensions:

            dataset = nested[
                dimensionality
            ]

            X_train = dataset.X[
                train_indices
            ]

            X_test = dataset.X[
                test_indices
            ]

            # ---------------------------------------------
            # Same MinMax preprocessing as baseline
            #
            # scaler is fitted only on X_train.
            # ---------------------------------------------

            (
                X_train_scaled,
                _,
                _,
            ) = scale_train_test(
                X_train,
                X_test,
            )

            # ---------------------------------------------
            # MLE-LID
            #
            # IMPORTANT:
            # No labels are supplied here.
            #
            # We estimate LID for ALL training observations,
            # because LID will later become part of the
            # unsupervised adaptive pairing mechanism.
            # ---------------------------------------------

            lid_values = estimate_lid(
                X_train_scaled,
                k=k,
            )

            # ---------------------------------------------
            # Safety checks
            # ---------------------------------------------

            if lid_values.shape != (
                train_indices.size,
            ):
                raise RuntimeError(
                    "Unexpected number of "
                    "LID estimates."
                )

            if not np.all(
                np.isfinite(
                    lid_values
                )
            ):
                raise RuntimeError(
                    "Non-finite LID values "
                    "were produced."
                )

            if np.any(
                lid_values <= 0
            ):
                raise RuntimeError(
                    "Non-positive LID values "
                    "were produced."
                )

            # ---------------------------------------------
            # Store point-level values
            # ---------------------------------------------

            for (
                train_position,
                lid_value,
            ) in enumerate(
                lid_values
            ):

                point_rows.append(
                    {
                        "dimensionality":
                            dimensionality,

                        "seed":
                            seed,

                        "k":
                            k,

                        "train_position":
                            int(
                                train_position
                            ),

                        "original_index":
                            int(
                                train_indices[
                                    train_position
                                ]
                            ),

                        "lid":
                            float(
                                lid_value
                            ),
                    }
                )

            # ---------------------------------------------
            # Distribution summary
            # ---------------------------------------------

            stats = summarize_lid(
                lid_values
            )

            summary_rows.append(
                {
                    "dimensionality":
                        dimensionality,

                    "seed":
                        seed,

                    "k":
                        k,

                    "n_train":
                        int(
                            train_indices.size
                        ),

                    **stats,
                }
            )

            # ---------------------------------------------
            # Checkpoint completed configurations
            # ---------------------------------------------

            _write_checkpoint(
                summary_rows,
                by_seed_path,
            )

            # ---------------------------------------------
            # Terminal progress
            # ---------------------------------------------

            print(
                f"completed "
                f"dimension={dimensionality:>3} "
                f"seed={seed} "
                f"median_lid="
                f"{stats['lid_median']:.4f} "
                f"mean_lid="
                f"{stats['lid_mean']:.4f}",
                flush=True,
            )

    # =====================================================
    # Final result tables
    # =====================================================

    point_results = pd.DataFrame(
        point_rows
    )

    by_seed = pd.DataFrame(
        summary_rows
    )

    summary = aggregate_by_dimension(
        by_seed
    )

    # -----------------------------------------------------
    # Save final CSVs
    # -----------------------------------------------------

    point_results.to_csv(
        points_path,
        index=False,
    )

    by_seed.to_csv(
        by_seed_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # =====================================================
    # Metadata
    # =====================================================

    metadata_payload = {

        "experiment":
            "MLE-LID analysis on ADERH "
            "distance-concentration baseline",

        "mode":
            prefix,

        "dimensions":
            list(dimensions),

        "seeds":
            list(seeds),

        "lid": {
            "estimator":
                "MLE-LID",

            "k":
                int(k),

            "distance":
                "euclidean",

            "population":
                "all scaled training observations",

            "labels_used_for_lid":
                False,

            "self_neighbor_excluded":
                True,
        },

        "dataset": {
            "generator":
                "aderh_distance.data_generation."
                "make_nested_datasets",

            "n_samples":
                2000,

            "contamination":
                0.05,

            "informative_dimensions":
                2,

            "normal_distribution":
                "N([0, 0], I)",

            "anomaly_distribution":
                "N([4, 4], I)",

            "irrelevant_distribution":
                "independent N(0, 1) "
                "for both classes",

            "nested_dimensions":
                True,
        },

        "split": {
            "function":
                "aderh_distance.experiments."
                "_split_indices",

            "method":
                "StratifiedShuffleSplit",

            "train_fraction":
                0.70,

            "test_fraction":
                0.30,

            "same_indices_across_dimensions_within_seed":
                True,

            "labels_used_only_for_split":
                True,
        },

        "preprocessing": {
            "function":
                "aderh_distance.experiments."
                "scale_train_test",

            "method":
                "MinMaxScaler",

            "fit_population":
                "training split only",
        },

        "environment": {
            "python":
                sys.version,

            "platform":
                platform.platform(),

            "packages": {
                name:
                    _package_version(
                        name
                    )

                for name in (
                    "numpy",
                    "pandas",
                    "scikit-learn",
                )
            },
        },

        "artifacts": {
            "point_level_lid":
                str(points_path),

            "by_seed":
                str(by_seed_path),

            "summary":
                str(summary_path),
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata_payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Final terminal output
    # =====================================================

    print(
        "\nLID baseline analysis completed."
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"- {points_path}"
    )

    print(
        f"- {by_seed_path}"
    )

    print(
        f"- {summary_path}"
    )

    print(
        f"- {metadata_path}"
    )

    return {
        "points":
            points_path,

        "by_seed":
            by_seed_path,

        "summary":
            summary_path,

        "metadata":
            metadata_path,
    }


# =========================================================
# Command-line interface
# =========================================================

def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "mode",
        choices=(
            "smoke",
            "full",
        ),
        help=(
            "smoke runs the same smoke configuration "
            "as the ADERH baseline; "
            "full runs all dimensions and seeds."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed used in smoke mode "
            "(default: 0)."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=(
            "Number of nearest neighbours "
            "for MLE-LID "
            "(default: 20)."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(
            "lid_analysis/results"
        ),
        help=(
            "Directory where LID results "
            "will be stored."
        ),
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:

    args = (
        _build_parser()
        .parse_args(argv)
    )

    if args.mode == "smoke":

        dimensions = (
            SMOKE_DIMENSIONS
        )

        seeds = (
            args.seed,
        )

    else:

        dimensions = (
            DEFAULT_DIMENSIONS
        )

        seeds = (
            FULL_SEEDS
        )

    artifacts = run_lid_analysis(
        dimensions=dimensions,
        seeds=seeds,
        k=args.k,
        prefix=args.mode,
        results_dir=args.results_dir,
    )

    print(
        "\nArtifacts:"
    )

    print(
        json.dumps(
            {
                key: str(value)
                for key, value
                in artifacts.items()
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )