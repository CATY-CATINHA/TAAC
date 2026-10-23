import numpy as np
import pandas as pd
import pytest

from aderh_distance.experiments import (
    aggregate_results,
    scale_train_test,
    score_diagnostics,
)


def test_minmax_scaler_is_fit_on_training_data_only() -> None:
    X_train = np.array([[0.0], [1.0]])
    X_test = np.array([[2.0]])

    train_scaled, test_scaled, scaler = scale_train_test(X_train, X_test)

    assert train_scaled[:, 0].tolist() == [0.0, 1.0]
    assert test_scaled[0, 0] == 2.0
    assert scaler.data_max_[0] == 1.0


def test_aggregation_uses_population_standard_deviation() -> None:
    rows = []
    for seed, value in enumerate((0.6, 0.8)):
        row = {
            "dimensionality": 2,
            "seed": seed,
            "n_train": 1_400,
            "n_test": 600,
            "test_anomaly_prevalence": 0.05,
        }
        for metric in (
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
        ):
            row[metric] = value
        rows.append(row)

    summary = aggregate_results(pd.DataFrame(rows)).iloc[0]

    assert summary["roc_auc_mean"] == pytest.approx(0.7)
    assert summary["roc_auc_std"] == pytest.approx(0.1)
    assert summary["n_runs"] == 2


def test_score_diagnostics_identifies_exact_uncovered_scores() -> None:
    scores = np.array([0.2, 1.0, 0.7, 1.0])
    labels = np.array([0, 0, 1, 1])

    result = score_diagnostics(scores, labels)

    assert result["test_unique_score_count"] == 3
    assert result["test_uncovered_fraction"] == pytest.approx(0.5)
    assert result["normal_uncovered_fraction"] == pytest.approx(0.5)
    assert result["anomaly_uncovered_fraction"] == pytest.approx(0.5)
    assert result["normal_score_mean"] == pytest.approx(0.6)
    assert result["anomaly_score_mean"] == pytest.approx(0.85)
    assert result["anomaly_normal_score_gap"] == pytest.approx(0.25)
