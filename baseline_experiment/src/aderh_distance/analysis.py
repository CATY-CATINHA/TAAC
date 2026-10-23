"""Reproducible post-run analysis of the full ADERH experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "dimensionality",
    "seed",
    "roc_auc",
    "average_precision",
    "relative_contrast",
    "n_test",
    "test_score_min",
    "test_score_max",
    "test_score_std",
    "test_unique_score_count",
    "test_uncovered_fraction",
    "normal_uncovered_fraction",
    "anomaly_uncovered_fraction",
    "anomaly_normal_score_gap",
}


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def _strictly_decreasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) < 0.0))


def _non_increasing(values: np.ndarray) -> bool:
    return bool(np.all(np.diff(values) <= 0.0))


def analyze_results(raw_results: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Analyze dimensional trends without refitting or using labels for tuning."""

    missing = REQUIRED_COLUMNS.difference(raw_results.columns)
    if missing:
        raise ValueError(f"raw results are missing columns: {sorted(missing)}")
    if raw_results.duplicated(["dimensionality", "seed"]).any():
        raise ValueError("raw results contain duplicate dimension/seed configurations")

    raw = raw_results.sort_values(["seed", "dimensionality"]).reset_index(drop=True)
    dimensions = sorted(int(value) for value in raw["dimensionality"].unique())
    seeds = sorted(int(value) for value in raw["seed"].unique())
    expected = pd.MultiIndex.from_product(
        [seeds, dimensions], names=["seed", "dimensionality"]
    )
    actual = pd.MultiIndex.from_frame(raw[["seed", "dimensionality"]])
    if not expected.equals(actual):
        raise ValueError("raw results do not form a complete seed/dimension grid")

    dimension_means = (
        raw.groupby("dimensionality", sort=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    seed_records: list[dict[str, object]] = []
    for seed, group in raw.groupby("seed", sort=True):
        ordered = group.sort_values("dimensionality")
        dimension = ordered["dimensionality"]
        contrast = ordered["relative_contrast"]
        roc_auc = ordered["roc_auc"]
        average_precision = ordered["average_precision"]
        seed_records.append(
            {
                "seed": int(seed),
                "relative_contrast_strictly_decreasing": _strictly_decreasing(
                    contrast.to_numpy()
                ),
                "roc_auc_non_increasing": _non_increasing(roc_auc.to_numpy()),
                "average_precision_non_increasing": _non_increasing(
                    average_precision.to_numpy()
                ),
                "dimension_vs_relative_contrast_spearman": _spearman(
                    dimension, contrast
                ),
                "dimension_vs_roc_auc_spearman": _spearman(dimension, roc_auc),
                "dimension_vs_average_precision_spearman": _spearman(
                    dimension, average_precision
                ),
                "relative_contrast_vs_roc_auc_spearman": _spearman(
                    contrast, roc_auc
                ),
                "relative_contrast_vs_average_precision_spearman": _spearman(
                    contrast, average_precision
                ),
            }
        )
    seed_trends = pd.DataFrame(seed_records)

    min_dimension = dimensions[0]
    max_dimension = dimensions[-1]
    first = dimension_means.loc[
        dimension_means["dimensionality"] == min_dimension
    ].iloc[0]
    last = dimension_means.loc[
        dimension_means["dimensionality"] == max_dimension
    ].iloc[0]
    max_rows = raw[raw["dimensionality"] == max_dimension]
    exact_baselines = (
        raw.assign(
            roc_at_baseline=np.isclose(raw["roc_auc"], 0.5, rtol=0.0, atol=1e-12),
            ap_at_baseline=np.isclose(
                raw["average_precision"],
                raw["test_anomaly_prevalence"],
                rtol=0.0,
                atol=1e-12,
            ),
        )
        .groupby("dimensionality", sort=True)[["roc_at_baseline", "ap_at_baseline"]]
        .sum()
        .astype(int)
    )
    threshold_rows = dimension_means[
        dimension_means["test_uncovered_fraction"] >= 0.95
    ]
    first_95 = (
        int(threshold_rows["dimensionality"].iloc[0])
        if not threshold_rows.empty
        else None
    )

    payload: dict[str, object] = {
        "scope": {
            "n_configurations": int(raw.shape[0]),
            "n_seeds": len(seeds),
            "dimensions": dimensions,
            "total_test_scores_at_max_dimension": int(max_rows["n_test"].sum()),
        },
        "endpoint_means": {
            "minimum_dimension": min_dimension,
            "maximum_dimension": max_dimension,
            "relative_contrast_start": float(first["relative_contrast"]),
            "relative_contrast_end": float(last["relative_contrast"]),
            "relative_contrast_fractional_decrease": float(
                1.0 - last["relative_contrast"] / first["relative_contrast"]
            ),
            "roc_auc_start": float(first["roc_auc"]),
            "roc_auc_end": float(last["roc_auc"]),
            "average_precision_start": float(first["average_precision"]),
            "average_precision_end": float(last["average_precision"]),
            "score_std_start": float(first["test_score_std"]),
            "score_std_end": float(last["test_score_std"]),
            "unique_scores_start": float(first["test_unique_score_count"]),
            "unique_scores_end": float(last["test_unique_score_count"]),
            "uncovered_fraction_start": float(first["test_uncovered_fraction"]),
            "uncovered_fraction_end": float(last["test_uncovered_fraction"]),
            "anomaly_normal_score_gap_start": float(
                first["anomaly_normal_score_gap"]
            ),
            "anomaly_normal_score_gap_end": float(last["anomaly_normal_score_gap"]),
        },
        "monotonicity_across_seeds": {
            "relative_contrast_strictly_decreasing_count": int(
                seed_trends["relative_contrast_strictly_decreasing"].sum()
            ),
            "roc_auc_non_increasing_count": int(
                seed_trends["roc_auc_non_increasing"].sum()
            ),
            "average_precision_non_increasing_count": int(
                seed_trends["average_precision_non_increasing"].sum()
            ),
            "denominator": len(seeds),
        },
        "dimension_mean_spearman": {
            "relative_contrast_vs_roc_auc": _spearman(
                dimension_means["relative_contrast"], dimension_means["roc_auc"]
            ),
            "relative_contrast_vs_average_precision": _spearman(
                dimension_means["relative_contrast"],
                dimension_means["average_precision"],
            ),
            "relative_contrast_vs_uncovered_fraction": _spearman(
                dimension_means["relative_contrast"],
                dimension_means["test_uncovered_fraction"],
            ),
        },
        "score_collapse": {
            "first_dimension_with_mean_uncovered_fraction_at_least_0_95": first_95,
            "all_max_dimension_scores_equal_one": bool(
                (max_rows["test_score_min"] == 1.0).all()
                and (max_rows["test_score_max"] == 1.0).all()
                and (max_rows["test_unique_score_count"] == 1).all()
            ),
            "max_dimension_normal_uncovered_fraction_mean": float(
                max_rows["normal_uncovered_fraction"].mean()
            ),
            "max_dimension_anomaly_uncovered_fraction_mean": float(
                max_rows["anomaly_uncovered_fraction"].mean()
            ),
        },
        "baseline_configuration_counts": {
            str(int(dimension)): {
                "roc_auc_equals_0_5": int(row["roc_at_baseline"]),
                "average_precision_equals_prevalence": int(row["ap_at_baseline"]),
            }
            for dimension, row in exact_baselines.iterrows()
        },
        "interpretation_guardrail": (
            "These results establish the failure mode for this controlled "
            "synthetic protocol and ADERH 1.0.0 defaults; they do not prove a "
            "universal failure for every dataset or hyperparameter setting."
        ),
    }
    return payload, seed_trends


def write_analysis(
    raw_path: Path,
    *,
    output_json: Path,
    output_seed_csv: Path,
) -> tuple[Path, Path]:
    raw = pd.read_csv(raw_path)
    analysis, seed_trends = analyze_results(raw)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_seed_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    seed_trends.to_csv(output_seed_csv, index=False)
    return output_json, output_seed_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("results/full_raw_results.csv"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("results/full_analysis.json")
    )
    parser.add_argument(
        "--output-seed-csv",
        type=Path,
        default=Path("results/full_seed_trends.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    json_path, seed_path = write_analysis(
        args.raw,
        output_json=args.output_json,
        output_seed_csv=args.output_seed_csv,
    )
    print(json.dumps({"analysis": str(json_path), "seed_trends": str(seed_path)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
