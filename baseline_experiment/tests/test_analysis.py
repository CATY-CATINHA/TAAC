import pandas as pd

from aderh_distance.analysis import analyze_results


def test_analysis_detects_monotonicity_and_complete_score_collapse() -> None:
    rows = []
    for seed in (0, 1):
        for dimension, contrast, roc, ap, uncovered, unique in (
            (2, 10.0, 1.0, 1.0, 0.0, 4),
            (5, 2.0, 0.8, 0.5, 0.5, 3),
            (10, 1.1, 0.5, 0.05, 1.0, 1),
        ):
            rows.append(
                {
                    "dimensionality": dimension,
                    "seed": seed,
                    "roc_auc": roc,
                    "average_precision": ap,
                    "relative_contrast": contrast,
                    "n_test": 4,
                    "test_anomaly_prevalence": 0.05,
                    "test_score_min": 1.0 if dimension == 10 else 0.1,
                    "test_score_max": 1.0,
                    "test_score_std": 0.0 if dimension == 10 else 0.2,
                    "test_unique_score_count": unique,
                    "test_uncovered_fraction": uncovered,
                    "normal_uncovered_fraction": uncovered,
                    "anomaly_uncovered_fraction": uncovered,
                    "anomaly_normal_score_gap": 0.0 if dimension == 10 else 0.4,
                }
            )

    analysis, seed_trends = analyze_results(pd.DataFrame(rows))

    assert analysis["scope"]["n_configurations"] == 6
    assert analysis["monotonicity_across_seeds"][
        "relative_contrast_strictly_decreasing_count"
    ] == 2
    assert analysis["score_collapse"]["all_max_dimension_scores_equal_one"]
    assert analysis["score_collapse"][
        "first_dimension_with_mean_uncovered_fraction_at_least_0_95"
    ] == 10
    assert seed_trends["roc_auc_non_increasing"].all()
