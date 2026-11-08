"""
Analyse the relationship between Local Intrinsic Dimensionality (LID)
and the high-dimensional failure mode observed in ADERH.

This script does NOT recompute LID or rerun ADERH.

Instead, it combines:

    lid_analysis/results/full_lid_by_seed.csv

with:

    baseline_experiment/results/full_raw_results.csv

using:

    dimensionality + seed

as the common experimental configuration.

The main relationships investigated are:

    LID <-> Relative Contrast
    LID <-> Uncovered Fraction
    LID <-> ROC-AUC
    LID <-> Average Precision

The main research question is:

    "When LID increases, does the geometry used by ADERH
    become less informative?"

Two complementary analyses are performed:

1. Spearman correlations inside each random seed
   across dimensionalities.

2. Spearman correlations using values aggregated
   across seeds for each dimensionality.

The script also generates the main relationship figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# =========================================================
# Configuration
# =========================================================

DEFAULT_LID_RESULTS = Path(
    "lid_analysis/results/full_lid_by_seed.csv"
)

DEFAULT_ADERH_RESULTS = Path(
    "baseline_experiment/results/full_raw_results.csv"
)

DEFAULT_RESULTS_DIR = Path(
    "lid_analysis/results/relationships"
)

DEFAULT_FIGURES_DIR = Path(
    "lid_analysis/figures/relationships"
)


# Metrics from the ADERH baseline that will be compared
# with LID.
ADERH_METRICS = (
    "relative_contrast",
    "uncovered_fraction",
    "roc_auc",
    "average_precision",
)


# LID statistics considered in the correlation analysis.
LID_METRICS = (
    "lid_mean",
    "lid_median",
)


# =========================================================
# Input validation
# =========================================================

def load_lid_results(path: Path) -> pd.DataFrame:
    """
    Load and validate LID results.

    Expected unit of observation:
        one row per dimensionality/seed configuration.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"LID results file not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {
        "dimensionality",
        "seed",
        "lid_mean",
        "lid_median",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns in LID results: "
            f"{sorted(missing)}"
        )

    if data.empty:
        raise ValueError(
            "LID results file is empty."
        )

    if data[
        ["dimensionality", "seed"]
    ].duplicated().any():
        raise ValueError(
            "Duplicated dimensionality/seed configurations "
            "found in LID results."
        )

    return data


def normalize_aderh_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize possible names used for the uncovered fraction.

    The ADERH baseline may store the metric as either:

        test_uncovered_fraction

    or:

        uncovered_fraction

    Internally this analysis always uses:

        uncovered_fraction
    """

    data = data.copy()

    if (
        "uncovered_fraction" not in data.columns
        and "test_uncovered_fraction" in data.columns
    ):
        data = data.rename(
            columns={
                "test_uncovered_fraction":
                    "uncovered_fraction"
            }
        )

    return data


def load_aderh_results(path: Path) -> pd.DataFrame:
    """
    Load and validate ADERH baseline results.

    Expected unit of observation:
        one row per dimensionality/seed configuration.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"ADERH results file not found: {path}"
        )

    data = pd.read_csv(path)

    data = normalize_aderh_columns(data)

    required_columns = {
        "dimensionality",
        "seed",
        "relative_contrast",
        "uncovered_fraction",
        "roc_auc",
        "average_precision",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns in ADERH results: "
            f"{sorted(missing)}\n"
            f"Available columns: {list(data.columns)}"
        )

    if data.empty:
        raise ValueError(
            "ADERH results file is empty."
        )

    if data[
        ["dimensionality", "seed"]
    ].duplicated().any():
        raise ValueError(
            "Duplicated dimensionality/seed configurations "
            "found in ADERH results."
        )

    return data


# =========================================================
# Merge LID and ADERH
# =========================================================

def merge_results(
    lid_results: pd.DataFrame,
    aderh_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge LID and ADERH results using dimensionality and seed.

    Only the variables required for the relationship analysis
    are retained.
    """

    lid_columns = [
        "dimensionality",
        "seed",
        "lid_mean",
        "lid_median",
    ]

    aderh_columns = [
        "dimensionality",
        "seed",
        "relative_contrast",
        "uncovered_fraction",
        "roc_auc",
        "average_precision",
    ]

    merged = pd.merge(
        lid_results[lid_columns],
        aderh_results[aderh_columns],
        on=["dimensionality", "seed"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "The merge produced no rows. "
            "Check dimensionality and seed columns."
        )

    # Check whether any experimental configurations
    # disappeared during the merge.
    expected_rows = min(
        len(lid_results),
        len(aderh_results),
    )

    if len(merged) != expected_rows:
        print(
            "\nWARNING:"
            "\nSome configurations were not matched during "
            "the LID/ADERH merge."
        )

    numeric_columns = [
        "lid_mean",
        "lid_median",
        *ADERH_METRICS,
    ]

    if not np.all(
        np.isfinite(
            merged[numeric_columns].to_numpy(
                dtype=float
            )
        )
    ):
        raise ValueError(
            "Merged results contain NaN or infinite values."
        )

    return merged.sort_values(
        ["seed", "dimensionality"]
    ).reset_index(drop=True)


# =========================================================
# Correlation analysis
# =========================================================

def safe_spearman(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """
    Compute a Spearman correlation with basic safety checks.

    Returns
    -------
    rho : float
        Spearman rank correlation coefficient.

    p_value : float
        Two-sided p-value returned by scipy.stats.spearmanr.
    """

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if x_array.size != y_array.size:
        raise ValueError(
            "Spearman inputs must have the same size."
        )

    if x_array.size < 3:
        raise ValueError(
            "At least three observations are required "
            "for the Spearman correlation."
        )

    if np.unique(x_array).size < 2:
        return np.nan, np.nan

    if np.unique(y_array).size < 2:
        return np.nan, np.nan

    result = spearmanr(
        x_array,
        y_array,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def correlations_by_seed(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate LID <-> ADERH Spearman correlations separately
    inside each random seed.

    Each seed contains the different dimensionalities.

    This checks whether the direction of each relationship is
    reproducible across random seeds.
    """

    records = []

    for seed, group in merged.groupby(
        "seed",
        sort=True,
    ):

        group = group.sort_values(
            "dimensionality"
        )

        for lid_metric in LID_METRICS:

            for aderh_metric in ADERH_METRICS:

                rho, p_value = safe_spearman(
                    group[lid_metric],
                    group[aderh_metric],
                )

                records.append(
                    {
                        "seed": int(seed),
                        "lid_metric": lid_metric,
                        "aderh_metric": aderh_metric,
                        "n_dimensions":
                            int(len(group)),
                        "spearman_rho": rho,
                        "p_value": p_value,
                    }
                )

    return pd.DataFrame(records)


# =========================================================
# Aggregation across seeds
# =========================================================

def aggregate_by_dimension(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate LID and ADERH metrics across random seeds
    for each dimensionality.

    These values are used for the main plots and for the
    dimension-level correlation analysis.
    """

    metrics = [
        "lid_mean",
        "lid_median",
        *ADERH_METRICS,
    ]

    records = []

    for dimensionality, group in merged.groupby(
        "dimensionality",
        sort=True,
    ):

        record = {
            "dimensionality":
                int(dimensionality),

            "n_seeds":
                int(group["seed"].nunique()),
        }

        for metric in metrics:

            values = group[metric].to_numpy(
                dtype=float
            )

            record[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            record[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=0,
                )
            )

        records.append(record)

    return pd.DataFrame(records)


def dimensional_correlations(
    by_dimension: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Spearman correlations on values aggregated
    across seeds for each dimensionality.
    """

    records = []

    for lid_metric in LID_METRICS:

        lid_column = f"{lid_metric}_mean"

        for aderh_metric in ADERH_METRICS:

            aderh_column = (
                f"{aderh_metric}_mean"
            )

            rho, p_value = safe_spearman(
                by_dimension[lid_column],
                by_dimension[aderh_column],
            )

            records.append(
                {
                    "lid_metric": lid_metric,
                    "aderh_metric":
                        aderh_metric,
                    "n_dimensions":
                        int(
                            len(by_dimension)
                        ),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(records)


# =========================================================
# Direction consistency across seeds
# =========================================================

def expected_direction(
    metric: str,
) -> str:
    """
    Return the hypothesised direction of the relationship.

    These directions are hypotheses, not enforced outcomes.

    LID ↑ -> Relative Contrast ↓
    LID ↑ -> Uncovered Fraction ↑
    LID ↑ -> ROC-AUC ↓
    LID ↑ -> Average Precision ↓
    """

    directions = {
        "relative_contrast": "negative",
        "uncovered_fraction": "positive",
        "roc_auc": "negative",
        "average_precision": "negative",
    }

    return directions[metric]


def summarize_seed_consistency(
    correlations: pd.DataFrame,
    lid_metric: str = "lid_mean",
) -> list[dict[str, object]]:
    """
    Summarize whether the sign of the correlation is consistent
    with the working hypothesis across seeds.
    """

    output = []

    selected = correlations[
        correlations["lid_metric"]
        == lid_metric
    ]

    for metric in ADERH_METRICS:

        group = selected[
            selected["aderh_metric"]
            == metric
        ]

        valid = group[
            np.isfinite(
                group["spearman_rho"]
            )
        ]

        expected = expected_direction(
            metric
        )

        if expected == "negative":
            matching = (
                valid["spearman_rho"] < 0
            )
        else:
            matching = (
                valid["spearman_rho"] > 0
            )

        output.append(
            {
                "aderh_metric": metric,
                "expected_direction":
                    expected,
                "n_valid_seeds":
                    int(len(valid)),
                "n_matching_expected_direction":
                    int(matching.sum()),
                "fraction_matching_expected_direction":
                    (
                        float(matching.mean())
                        if len(valid) > 0
                        else None
                    ),
                "mean_seed_spearman_rho":
                    (
                        float(
                            valid[
                                "spearman_rho"
                            ].mean()
                        )
                        if len(valid) > 0
                        else None
                    ),
                "median_seed_spearman_rho":
                    (
                        float(
                            valid[
                                "spearman_rho"
                            ].median()
                        )
                        if len(valid) > 0
                        else None
                    ),
            }
        )

    return output


# =========================================================
# Plotting
# =========================================================

def pretty_metric_name(
    metric: str,
) -> str:
    """
    Human-readable metric names for figure labels.
    """

    labels = {
        "relative_contrast":
            "Relative Contrast",

        "uncovered_fraction":
            "Uncovered Fraction",

        "roc_auc":
            "ROC-AUC",

        "average_precision":
            "Average Precision",
    }

    return labels[metric]


def plot_relationship(
    by_dimension: pd.DataFrame,
    aderh_metric: str,
    figures_dir: Path,
) -> Path:
    """
    Plot Mean LID against one ADERH metric.

    Each point represents the mean across random seeds for
    one dimensionality.

    Dimensionality labels are added to each point.
    """

    x = by_dimension[
        "lid_mean_mean"
    ].to_numpy(dtype=float)

    y = by_dimension[
        f"{aderh_metric}_mean"
    ].to_numpy(dtype=float)

    dimensions = by_dimension[
        "dimensionality"
    ].to_numpy(dtype=int)

    rho, _ = safe_spearman(
        x,
        y,
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    ax.plot(
        x,
        y,
        marker="o",
    )

    for x_value, y_value, dimension in zip(
        x,
        y,
        dimensions,
    ):
        ax.annotate(
            f"{dimension}D",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(
        "Mean LID"
    )

    ax.set_ylabel(
        pretty_metric_name(
            aderh_metric
        )
    )

    ax.set_title(
        "Mean LID vs "
        f"{pretty_metric_name(aderh_metric)}"
        "\n"
        f"Spearman ρ = {rho:.3f}"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figures_dir
        / f"lid_vs_{aderh_metric}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def create_relationship_figures(
    by_dimension: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, Path]:
    """
    Generate the four main LID <-> ADERH figures.
    """

    paths = {}

    for metric in ADERH_METRICS:

        paths[metric] = plot_relationship(
            by_dimension=by_dimension,
            aderh_metric=metric,
            figures_dir=figures_dir,
        )

    return paths


# =========================================================
# Summary
# =========================================================

def build_summary(
    merged: pd.DataFrame,
    by_dimension: pd.DataFrame,
    by_seed_correlations: pd.DataFrame,
    dimensional_corr: pd.DataFrame,
) -> dict[str, object]:
    """
    Create a machine-readable summary of the experiment.
    """

    mean_lid_corr = dimensional_corr[
        dimensional_corr["lid_metric"]
        == "lid_mean"
    ]

    dimensional_results = {}

    for _, row in mean_lid_corr.iterrows():

        dimensional_results[
            row["aderh_metric"]
        ] = {
            "spearman_rho":
                float(
                    row["spearman_rho"]
                ),
            "p_value":
                float(
                    row["p_value"]
                ),
        }

    return {
        "research_question":
            (
                "When LID increases, does the geometry "
                "used by ADERH become less informative?"
            ),

        "analysis_type":
            (
                "Association analysis between MLE-LID "
                "and ADERH baseline metrics"
            ),

        "n_configurations":
            int(len(merged)),

        "n_dimensions":
            int(
                merged[
                    "dimensionality"
                ].nunique()
            ),

        "n_seeds":
            int(
                merged["seed"].nunique()
            ),

        "lid_primary_metric":
            "lid_mean",

        "lid_secondary_metric":
            "lid_median",

        "correlation":
            "Spearman rank correlation",

        "dimensional_correlations":
            dimensional_results,

        "seed_direction_consistency":
            summarize_seed_consistency(
                by_seed_correlations,
                lid_metric="lid_mean",
            ),

        "important_interpretation_note":
            (
                "Dimensionality is a common driver of both "
                "LID and ADERH degradation in this controlled "
                "experiment. Therefore, these correlations "
                "show association and consistency of trends, "
                "not a causal effect of LID on ADERH."
            ),

        "dimensions":
            [
                int(value)
                for value in by_dimension[
                    "dimensionality"
                ]
            ],
    }


# =========================================================
# Main experiment
# =========================================================

def run_relationship_analysis(
    *,
    lid_results_path: Path,
    aderh_results_path: Path,
    results_dir: Path,
    figures_dir: Path,
) -> dict[str, Path]:
    """
    Run the complete LID <-> ADERH relationship analysis.
    """

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load existing experiment results
    # -----------------------------------------------------

    lid_results = load_lid_results(
        lid_results_path
    )

    aderh_results = load_aderh_results(
        aderh_results_path
    )

    print(
        f"LID configurations: "
        f"{len(lid_results)}"
    )

    print(
        f"ADERH configurations: "
        f"{len(aderh_results)}"
    )

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    merged = merge_results(
        lid_results,
        aderh_results,
    )

    print(
        f"Merged configurations: "
        f"{len(merged)}"
    )

    # -----------------------------------------------------
    # Correlation analyses
    # -----------------------------------------------------

    by_seed_corr = correlations_by_seed(
        merged
    )

    by_dimension = aggregate_by_dimension(
        merged
    )

    dimensional_corr = dimensional_correlations(
        by_dimension
    )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------

    merged_path = (
        results_dir
        / "lid_aderh_merged.csv"
    )

    by_dimension_path = (
        results_dir
        / "relationship_by_dimension.csv"
    )

    seed_corr_path = (
        results_dir
        / "correlations_by_seed.csv"
    )

    dimensional_corr_path = (
        results_dir
        / "correlations_by_dimension.csv"
    )

    summary_path = (
        results_dir
        / "relationship_summary.json"
    )

    # -----------------------------------------------------
    # Save tables
    # -----------------------------------------------------

    merged.to_csv(
        merged_path,
        index=False,
    )

    by_dimension.to_csv(
        by_dimension_path,
        index=False,
    )

    by_seed_corr.to_csv(
        seed_corr_path,
        index=False,
    )

    dimensional_corr.to_csv(
        dimensional_corr_path,
        index=False,
    )

    # -----------------------------------------------------
    # Figures
    # -----------------------------------------------------

    figure_paths = create_relationship_figures(
        by_dimension,
        figures_dir,
    )

    # -----------------------------------------------------
    # JSON summary
    # -----------------------------------------------------

    summary = build_summary(
        merged=merged,
        by_dimension=by_dimension,
        by_seed_correlations=by_seed_corr,
        dimensional_corr=dimensional_corr,
    )

    summary["artifacts"] = {
        "merged":
            str(merged_path),

        "by_dimension":
            str(by_dimension_path),

        "correlations_by_seed":
            str(seed_corr_path),

        "correlations_by_dimension":
            str(dimensional_corr_path),

        "figures": {
            key: str(value)
            for key, value
            in figure_paths.items()
        },
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------

    print(
        "\nLID <-> ADERH relationship analysis completed."
    )

    print(
        "\nSpearman correlations using "
        "dimension-level means:"
    )

    primary = dimensional_corr[
        dimensional_corr["lid_metric"]
        == "lid_mean"
    ]

    for _, row in primary.iterrows():

        print(
            f"- Mean LID vs "
            f"{pretty_metric_name(row['aderh_metric'])}: "
            f"rho={row['spearman_rho']:.4f}, "
            f"p={row['p_value']:.4g}"
        )

    print(
        "\nDirection consistency across seeds:"
    )

    consistency = summarize_seed_consistency(
        by_seed_corr,
        lid_metric="lid_mean",
    )

    for result in consistency:

        print(
            f"- {pretty_metric_name(result['aderh_metric'])}: "
            f"{result['n_matching_expected_direction']}"
            f"/{result['n_valid_seeds']} seeds "
            f"match the hypothesised "
            f"{result['expected_direction']} direction"
        )

    print(
        "\nGenerated files:"
    )

    artifacts = {
        "merged":
            merged_path,

        "by_dimension":
            by_dimension_path,

        "correlations_by_seed":
            seed_corr_path,

        "correlations_by_dimension":
            dimensional_corr_path,

        "summary":
            summary_path,

        **{
            f"figure_{key}": value
            for key, value
            in figure_paths.items()
        },
    }

    for key, path in artifacts.items():

        print(
            f"- {key}: {path}"
        )

    return artifacts


# =========================================================
# Command-line interface
# =========================================================

def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--lid-results",
        type=Path,
        default=DEFAULT_LID_RESULTS,
        help=(
            "Path to full_lid_by_seed.csv."
        ),
    )

    parser.add_argument(
        "--aderh-results",
        type=Path,
        default=DEFAULT_ADERH_RESULTS,
        help=(
            "Path to the ADERH "
            "full_raw_results.csv file."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory used for relationship "
            "analysis result files."
        ),
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=(
            "Directory used for relationship figures."
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

    run_relationship_analysis(
        lid_results_path=args.lid_results,
        aderh_results_path=args.aderh_results,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )