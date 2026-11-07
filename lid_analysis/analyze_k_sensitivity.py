"""
Sensitivity analysis of MLE-LID with respect to neighbourhood size k.

This experiment evaluates whether the conclusions obtained from the
LID <-> ADERH relationship analysis remain stable when the number of
nearest neighbours used by MLE-LID changes.

The default values are:

    k = 5, 10, 20, 30

Important methodological point:
This is a sensitivity / robustness analysis.

The objective is NOT to select the k that maximizes correlation with
ADERH performance.

Instead, the objective is to determine whether the observed
relationships remain qualitatively stable across reasonable
neighbourhood sizes.

For each k, the analysis evaluates:

    LID <-> Relative Contrast
    LID <-> Uncovered Fraction
    LID <-> ROC-AUC
    LID <-> Average Precision

The existing k=20 results are reused by default to avoid unnecessary
recomputation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aderh_distance.data_generation import DEFAULT_DIMENSIONS
from aderh_distance.experiments import FULL_SEEDS

from lid_analysis.analyze_lid_baseline import (
    run_lid_analysis,
)

from lid_analysis.analyze_lid_aderh_relationship import (
    load_lid_results,
    load_aderh_results,
    merge_results,
    correlations_by_seed,
    aggregate_by_dimension,
    dimensional_correlations,
    summarize_seed_consistency,
    expected_direction,
    pretty_metric_name,
)


# =========================================================
# Configuration
# =========================================================

DEFAULT_K_VALUES = (
    5,
    10,
    20,
    30,
)

REFERENCE_K = 20


# Existing k=20 experiment already generated during
# the main LID analysis.
DEFAULT_K20_RESULTS = Path(
    "lid_analysis/results/full_lid_by_seed.csv"
)


# Original ADERH baseline results.
DEFAULT_ADERH_RESULTS = Path(
    "baseline_experiment/results/full_raw_results.csv"
)


# New sensitivity-analysis outputs.
DEFAULT_RESULTS_DIR = Path(
    "lid_analysis/results/k_sensitivity"
)

DEFAULT_FIGURES_DIR = Path(
    "lid_analysis/figures/k_sensitivity"
)


ADERH_METRICS = (
    "relative_contrast",
    "uncovered_fraction",
    "roc_auc",
    "average_precision",
)


# =========================================================
# LID experiment for each k
# =========================================================

def get_lid_results_for_k(
    *,
    k: int,
    results_dir: Path,
    existing_k20_path: Path,
    force_recompute_k20: bool,
) -> tuple[pd.DataFrame, str]:
    """
    Obtain LID results for one neighbourhood size k.

    For k=20, the already existing main experiment is reused
    unless force_recompute_k20=True.

    For all other k values, the complete LID experiment is run
    using exactly the same dimensions, seeds, splits and
    preprocessing as the original baseline.

    Returns
    -------
    lid_results : pandas.DataFrame
        One row per dimensionality/seed configuration.

    source : str
        Description of where the results came from.
    """

    # -----------------------------------------------------
    # Reuse existing k=20 experiment
    # -----------------------------------------------------

    if (
        k == REFERENCE_K
        and not force_recompute_k20
        and existing_k20_path.exists()
    ):
        print(
            "\n"
            f"k={k}: reusing existing LID results:"
            f"\n{existing_k20_path}"
        )

        data = load_lid_results(
            existing_k20_path
        )

        return (
            data,
            str(existing_k20_path),
        )

    # -----------------------------------------------------
    # Run new LID experiment
    # -----------------------------------------------------

    print(
        "\n"
        f"k={k}: running complete LID experiment..."
    )

    artifacts = run_lid_analysis(
        dimensions=DEFAULT_DIMENSIONS,
        seeds=FULL_SEEDS,
        k=k,
        prefix=f"full_k{k}",
        results_dir=results_dir,
    )

    by_seed_path = artifacts[
        "by_seed"
    ]

    data = load_lid_results(
        by_seed_path
    )

    return (
        data,
        str(by_seed_path),
    )


# =========================================================
# Analysis for one k
# =========================================================

def analyse_single_k(
    *,
    k: int,
    lid_results: pd.DataFrame,
    aderh_results: pd.DataFrame,
) -> dict[str, object]:
    """
    Perform the complete LID <-> ADERH analysis for one k.
    """

    # -----------------------------------------------------
    # Merge LID with ADERH
    # -----------------------------------------------------

    merged = merge_results(
        lid_results,
        aderh_results,
    )

    merged.insert(
        0,
        "k",
        k,
    )

    # -----------------------------------------------------
    # Correlations inside each seed
    # -----------------------------------------------------

    merged_without_k = merged.drop(
        columns="k"
    )

    seed_correlations = correlations_by_seed(
        merged_without_k
    )

    seed_correlations.insert(
        0,
        "k",
        k,
    )

    # -----------------------------------------------------
    # Aggregate across seeds for each dimensionality
    # -----------------------------------------------------

    by_dimension = aggregate_by_dimension(
        merged_without_k
    )

    by_dimension.insert(
        0,
        "k",
        k,
    )

    # -----------------------------------------------------
    # Dimension-level correlations
    # -----------------------------------------------------

    dimension_correlations = (
        dimensional_correlations(
            by_dimension.drop(
                columns="k"
            )
        )
    )

    dimension_correlations.insert(
        0,
        "k",
        k,
    )

    # -----------------------------------------------------
    # Direction consistency across seeds
    # -----------------------------------------------------

    consistency = summarize_seed_consistency(
        seed_correlations.drop(
            columns="k"
        ),
        lid_metric="lid_mean",
    )

    consistency_rows = []

    for row in consistency:

        consistency_rows.append(
            {
                "k": k,
                **row,
            }
        )

    consistency_frame = pd.DataFrame(
        consistency_rows
    )

    return {
        "merged": merged,
        "by_dimension": by_dimension,
        "seed_correlations":
            seed_correlations,
        "dimension_correlations":
            dimension_correlations,
        "consistency":
            consistency_frame,
    }


# =========================================================
# Plot 1:
# Dimensionality vs Mean LID for every k
# =========================================================

def plot_lid_by_k(
    by_dimension: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Plot mean LID against dimensionality for each k.

    This shows how the absolute LID estimate changes with
    neighbourhood size, and whether the increasing trend
    with dimensionality remains stable.
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for k, group in by_dimension.groupby(
        "k",
        sort=True,
    ):

        group = group.sort_values(
            "dimensionality"
        )

        ax.plot(
            group["dimensionality"],
            group["lid_mean_mean"],
            marker="o",
            label=f"k={k}",
        )

    ax.set_xlabel(
        "Dimensionality"
    )

    ax.set_ylabel(
        "Mean LID"
    )

    ax.set_title(
        "MLE-LID Sensitivity to Neighbourhood Size"
    )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Neighbourhood size"
    )

    fig.tight_layout()

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figures_dir
        / "lid_by_k_vs_dimensionality.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# =========================================================
# Plot 2:
# Correlations vs k
# =========================================================

def plot_correlations_by_k(
    correlations: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """
    Plot dimension-level Spearman correlations for each k.

    Only Mean LID is shown in the primary sensitivity figure.
    """

    selected = correlations[
        correlations["lid_metric"]
        == "lid_mean"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for metric in ADERH_METRICS:

        group = selected[
            selected["aderh_metric"]
            == metric
        ].sort_values(
            "k"
        )

        ax.plot(
            group["k"],
            group["spearman_rho"],
            marker="o",
            label=pretty_metric_name(
                metric
            ),
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Number of Neighbours (k)"
    )

    ax.set_ylabel(
        "Spearman ρ"
    )

    ax.set_title(
        "Stability of LID–ADERH Relationships Across k"
    )

    ax.set_xticks(
        sorted(
            selected["k"].unique()
        )
    )

    ax.set_ylim(
        -1.1,
        1.1,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        figures_dir
        / "correlations_by_k.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# =========================================================
# Robustness summary
# =========================================================

def build_robustness_table(
    dimension_correlations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact robustness table for Mean LID.

    One row is produced per k and ADERH metric.

    The table records whether the observed correlation sign
    matches the expected direction.
    """

    selected = dimension_correlations[
        dimension_correlations[
            "lid_metric"
        ] == "lid_mean"
    ].copy()

    records = []

    for _, row in selected.iterrows():

        metric = row[
            "aderh_metric"
        ]

        rho = float(
            row["spearman_rho"]
        )

        expected = expected_direction(
            metric
        )

        if expected == "negative":

            matches = (
                rho < 0
            )

        else:

            matches = (
                rho > 0
            )

        records.append(
            {
                "k":
                    int(row["k"]),

                "aderh_metric":
                    metric,

                "expected_direction":
                    expected,

                "spearman_rho":
                    rho,

                "p_value":
                    float(
                        row["p_value"]
                    ),

                "matches_expected_direction":
                    bool(matches),
            }
        )

    return pd.DataFrame(
        records
    ).sort_values(
        [
            "k",
            "aderh_metric",
        ]
    ).reset_index(
        drop=True
    )


# =========================================================
# JSON summary
# =========================================================

def build_summary(
    *,
    k_values: tuple[int, ...],
    robustness: pd.DataFrame,
    consistency: pd.DataFrame,
    sources: dict[int, str],
) -> dict[str, object]:
    """
    Build the final machine-readable sensitivity summary.
    """

    metric_summary = {}

    for metric in ADERH_METRICS:

        metric_rows = robustness[
            robustness["aderh_metric"]
            == metric
        ]

        seed_rows = consistency[
            consistency["aderh_metric"]
            == metric
        ]

        metric_summary[
            metric
        ] = {
            "expected_direction":
                expected_direction(
                    metric
                ),

            "rho_by_k": {
                str(
                    int(row["k"])
                ):
                    float(
                        row["spearman_rho"]
                    )

                for _, row
                in metric_rows.iterrows()
            },

            "direction_stable_across_all_k":
                bool(
                    metric_rows[
                        "matches_expected_direction"
                    ].all()
                ),

            "all_seeds_match_expected_direction_for_all_k":
                bool(
                    (
                        seed_rows[
                            "fraction_matching_expected_direction"
                        ]
                        == 1.0
                    ).all()
                ),
        }

    overall_direction_stability = bool(
        robustness[
            "matches_expected_direction"
        ].all()
    )

    return {
        "experiment":
            "MLE-LID k sensitivity analysis",

        "objective":
            (
                "Determine whether the relationship between "
                "LID and ADERH degradation is robust to the "
                "choice of neighbourhood size k."
            ),

        "k_values":
            list(k_values),

        "reference_k":
            REFERENCE_K,

        "selection_policy":
            (
                "k is not selected according to the strongest "
                "correlation with ADERH. This experiment is a "
                "sensitivity analysis, not performance tuning."
            ),

        "dimensions":
            [
                int(value)
                for value in DEFAULT_DIMENSIONS
            ],

        "seeds":
            [
                int(value)
                for value in FULL_SEEDS
            ],

        "lid_sources": {
            str(k):
                source

            for k, source
            in sources.items()
        },

        "metric_results":
            metric_summary,

        "overall_direction_stable_across_all_k":
            overall_direction_stability,

        "interpretation_note":
            (
                "Stable correlation directions across k indicate "
                "that the conclusions are not an artefact of the "
                "specific reference choice k=20. Correlation "
                "strength should not be used to select k."
            ),
    }


# =========================================================
# Complete sensitivity analysis
# =========================================================

def run_k_sensitivity(
    *,
    k_values: tuple[int, ...],
    existing_k20_path: Path,
    aderh_results_path: Path,
    results_dir: Path,
    figures_dir: Path,
    force_recompute_k20: bool = False,
) -> dict[str, Path]:
    """
    Run the complete MLE-LID neighbourhood-size
    sensitivity analysis.
    """

    if not k_values:
        raise ValueError(
            "At least one k value must be provided."
        )

    if any(
        k <= 1
        for k in k_values
    ):
        raise ValueError(
            "All k values must be greater than 1."
        )

    if len(
        set(k_values)
    ) != len(
        k_values
    ):
        raise ValueError(
            "k values must not contain duplicates."
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load ADERH once.
    #
    # ADERH itself does NOT change with k.
    # -----------------------------------------------------

    aderh_results = load_aderh_results(
        aderh_results_path
    )

    print(
        "\nADERH baseline configurations: "
        f"{len(aderh_results)}"
    )

    # -----------------------------------------------------
    # Collect analyses for every k
    # -----------------------------------------------------

    all_lid = []
    all_merged = []
    all_by_dimension = []
    all_seed_correlations = []
    all_dimension_correlations = []
    all_consistency = []

    sources = {}

    for k in k_values:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"ANALYSING k={k}"
        )

        print(
            "=" * 60
        )

        # -------------------------------------------------
        # Obtain LID results
        # -------------------------------------------------

        lid_results, source = (
            get_lid_results_for_k(
                k=k,
                results_dir=results_dir,
                existing_k20_path=
                    existing_k20_path,
                force_recompute_k20=
                    force_recompute_k20,
            )
        )

        sources[k] = source

        lid_copy = lid_results.copy()

        # Ensure the correct k is recorded.
        if "k" in lid_copy.columns:

            lid_copy["k"] = k

        else:

            lid_copy.insert(
                2,
                "k",
                k,
            )

        all_lid.append(
            lid_copy
        )

        # -------------------------------------------------
        # LID <-> ADERH analysis
        # -------------------------------------------------

        analysis = analyse_single_k(
            k=k,
            lid_results=lid_results,
            aderh_results=aderh_results,
        )

        all_merged.append(
            analysis["merged"]
        )

        all_by_dimension.append(
            analysis["by_dimension"]
        )

        all_seed_correlations.append(
            analysis[
                "seed_correlations"
            ]
        )

        all_dimension_correlations.append(
            analysis[
                "dimension_correlations"
            ]
        )

        all_consistency.append(
            analysis["consistency"]
        )

    # =====================================================
    # Combine every k
    # =====================================================

    lid_all_k = pd.concat(
        all_lid,
        ignore_index=True,
    )

    merged_all_k = pd.concat(
        all_merged,
        ignore_index=True,
    )

    by_dimension_all_k = pd.concat(
        all_by_dimension,
        ignore_index=True,
    )

    seed_correlations_all_k = pd.concat(
        all_seed_correlations,
        ignore_index=True,
    )

    dimension_correlations_all_k = pd.concat(
        all_dimension_correlations,
        ignore_index=True,
    )

    consistency_all_k = pd.concat(
        all_consistency,
        ignore_index=True,
    )

    # =====================================================
    # Robustness table
    # =====================================================

    robustness = build_robustness_table(
        dimension_correlations_all_k
    )

    # =====================================================
    # Output paths
    # =====================================================

    lid_all_k_path = (
        results_dir
        / "k_sensitivity_lid_by_seed.csv"
    )

    merged_all_k_path = (
        results_dir
        / "k_sensitivity_merged.csv"
    )

    by_dimension_path = (
        results_dir
        / "k_sensitivity_by_dimension.csv"
    )

    seed_corr_path = (
        results_dir
        / "k_sensitivity_correlations_by_seed.csv"
    )

    dimension_corr_path = (
        results_dir
        / "k_sensitivity_correlations.csv"
    )

    consistency_path = (
        results_dir
        / "k_direction_consistency.csv"
    )

    robustness_path = (
        results_dir
        / "k_robustness_summary.csv"
    )

    summary_path = (
        results_dir
        / "k_sensitivity_summary.json"
    )

    # =====================================================
    # Save CSV outputs
    # =====================================================

    lid_all_k.to_csv(
        lid_all_k_path,
        index=False,
    )

    merged_all_k.to_csv(
        merged_all_k_path,
        index=False,
    )

    by_dimension_all_k.to_csv(
        by_dimension_path,
        index=False,
    )

    seed_correlations_all_k.to_csv(
        seed_corr_path,
        index=False,
    )

    dimension_correlations_all_k.to_csv(
        dimension_corr_path,
        index=False,
    )

    consistency_all_k.to_csv(
        consistency_path,
        index=False,
    )

    robustness.to_csv(
        robustness_path,
        index=False,
    )

    # =====================================================
    # Figures
    # =====================================================

    lid_figure = plot_lid_by_k(
        by_dimension_all_k,
        figures_dir,
    )

    correlation_figure = (
        plot_correlations_by_k(
            dimension_correlations_all_k,
            figures_dir,
        )
    )

    # =====================================================
    # JSON summary
    # =====================================================

    summary = build_summary(
        k_values=k_values,
        robustness=robustness,
        consistency=consistency_all_k,
        sources=sources,
    )

    summary["artifacts"] = {
        "lid_by_seed":
            str(lid_all_k_path),

        "merged":
            str(merged_all_k_path),

        "by_dimension":
            str(by_dimension_path),

        "correlations_by_seed":
            str(seed_corr_path),

        "correlations":
            str(dimension_corr_path),

        "direction_consistency":
            str(consistency_path),

        "robustness":
            str(robustness_path),

        "lid_figure":
            str(lid_figure),

        "correlation_figure":
            str(correlation_figure),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # Terminal output
    # =====================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "K SENSITIVITY ANALYSIS COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        "\nDimension-level Spearman correlations "
        "using Mean LID:"
    )

    primary = (
        dimension_correlations_all_k[
            dimension_correlations_all_k[
                "lid_metric"
            ] == "lid_mean"
        ]
        .sort_values(
            [
                "k",
                "aderh_metric",
            ]
        )
    )

    for k in sorted(
        primary["k"].unique()
    ):

        print(
            f"\nk={k}"
        )

        group = primary[
            primary["k"] == k
        ]

        for _, row in group.iterrows():

            print(
                "  "
                f"{pretty_metric_name(row['aderh_metric'])}: "
                f"rho={row['spearman_rho']:.4f}"
            )

    print(
        "\nDirection robustness:"
    )

    for metric in ADERH_METRICS:

        group = robustness[
            robustness["aderh_metric"]
            == metric
        ]

        n_matching = int(
            group[
                "matches_expected_direction"
            ].sum()
        )

        print(
            f"- {pretty_metric_name(metric)}: "
            f"{n_matching}/{len(group)} k values "
            "match the expected direction"
        )

    print(
        "\nOverall direction stable across all k: "
        f"{robustness['matches_expected_direction'].all()}"
    )

    print(
        "\nGenerated files:"
    )

    artifacts = {
        "lid_by_seed":
            lid_all_k_path,

        "merged":
            merged_all_k_path,

        "by_dimension":
            by_dimension_path,

        "correlations_by_seed":
            seed_corr_path,

        "correlations":
            dimension_corr_path,

        "direction_consistency":
            consistency_path,

        "robustness":
            robustness_path,

        "summary":
            summary_path,

        "figure_lid":
            lid_figure,

        "figure_correlations":
            correlation_figure,
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
        "--ks",
        type=int,
        nargs="+",
        default=list(
            DEFAULT_K_VALUES
        ),
        help=(
            "Neighbourhood sizes to test. "
            "Default: 5 10 20 30"
        ),
    )

    parser.add_argument(
        "--existing-k20",
        type=Path,
        default=DEFAULT_K20_RESULTS,
        help=(
            "Existing k=20 LID results."
        ),
    )

    parser.add_argument(
        "--aderh-results",
        type=Path,
        default=DEFAULT_ADERH_RESULTS,
        help=(
            "ADERH baseline full_raw_results.csv."
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Output directory for sensitivity results."
        ),
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=(
            "Output directory for sensitivity figures."
        ),
    )

    parser.add_argument(
        "--force-recompute-k20",
        action="store_true",
        help=(
            "Recompute k=20 instead of reusing the "
            "existing full LID results."
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

    k_values = tuple(
        int(k)
        for k in args.ks
    )

    run_k_sensitivity(
        k_values=k_values,
        existing_k20_path=
            args.existing_k20,
        aderh_results_path=
            args.aderh_results,
        results_dir=
            args.results_dir,
        figures_dir=
            args.figures_dir,
        force_recompute_k20=
            args.force_recompute_k20,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )