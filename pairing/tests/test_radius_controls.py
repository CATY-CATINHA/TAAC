import numpy as np
import pytest
from sklearn.base import clone

from pairing.common.coverage import (
    coverage_at_scale,
    coverage_scale_thresholds,
    scale_for_target_coverage,
)
from pairing.random_pairing.corrected_random_aderh import CorrectedRandomADERH
from pairing.random_pairing.radius_controls import (
    CoverageCalibratedRandomADERH,
    ShuffledCoverageCalibratedRandomADERH,
    ShuffledRadiusRandomADERH,
)


def test_coverage_scale_is_selected_from_empirical_thresholds():
    X = np.array([[0.0], [1.0], [2.0]])
    centers = np.array([[[0.0]]])
    squared_radii = np.array([[1.0]])

    thresholds = coverage_scale_thresholds(
        X,
        centers,
        squared_radii,
        point_indices=np.arange(3),
        estimator_indices=np.array([0]),
    )

    assert np.array_equal(thresholds, np.array([0.0, 1.0, 4.0]))
    gamma = scale_for_target_coverage(thresholds, 2 / 3)
    assert gamma == pytest.approx(1.0)
    assert coverage_at_scale(thresholds, gamma) == pytest.approx(2 / 3)


def test_shuffled_radius_preserves_centers_and_radius_distribution():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(100, 6))
    corrected = CorrectedRandomADERH(
        n_estimators=8,
        n=12,
        random_state=4,
    ).fit(X)
    shuffled = ShuffledRadiusRandomADERH(
        n_estimators=8,
        n=12,
        random_state=4,
    ).fit(X)

    assert np.array_equal(corrected._centers, shuffled._centers)

    corrected_pairs = corrected._sq_radii[:, : corrected.n]
    shuffled_pairs = shuffled._sq_radii[:, : shuffled.n]
    assert np.allclose(
        np.sort(corrected_pairs, axis=1),
        np.sort(shuffled_pairs, axis=1),
    )
    assert not np.array_equal(corrected_pairs, shuffled_pairs)
    assert np.array_equal(
        shuffled._sq_radii[:, : shuffled.n],
        shuffled._sq_radii[:, shuffled.n :],
    )


@pytest.mark.parametrize(
    "Model",
    [
        CoverageCalibratedRandomADERH,
        ShuffledCoverageCalibratedRandomADERH,
    ],
)
def test_coverage_calibration_matches_original_training_target(Model):
    rng = np.random.default_rng(22)
    X = rng.normal(size=(100, 6))
    model = Model(
        n_estimators=8,
        n=12,
        random_state=5,
        calibration_max_points=60,
        calibration_max_estimators=8,
    ).fit(X)

    tolerance = 1 / (60 * 8)
    assert np.isfinite(model.radius_scale_gamma_)
    assert model.radius_scale_gamma_ >= 1
    assert model.calibration_point_count_ == 60
    assert model.calibration_estimator_count_ == 8
    assert (
        model.calibration_achieved_coverage_
        >= model.calibration_target_coverage_
    )
    if model.calibration_scale_floor_applied_:
        assert model.radius_scale_gamma_ == 1.0
        assert model.calibration_base_coverage_ >= (
            model.calibration_target_coverage_
        )
        assert model.calibration_achieved_coverage_ == pytest.approx(
            model.calibration_base_coverage_
        )
    else:
        assert (
            model.calibration_achieved_coverage_
            - model.calibration_target_coverage_
            <= tolerance + 1e-12
        )
    assert np.all(np.isfinite(model.decision_scores_))


@pytest.mark.parametrize(
    "Model",
    [
        CoverageCalibratedRandomADERH,
        ShuffledCoverageCalibratedRandomADERH,
    ],
)
def test_coverage_calibrated_models_are_cloneable(Model):
    model = Model(
        calibration_max_points=40,
        calibration_max_estimators=7,
        random_state=6,
    )
    cloned = clone(model)

    assert type(cloned) is Model
    assert cloned.calibration_max_points == 40
    assert cloned.calibration_max_estimators == 7


def test_invalid_calibration_sample_size_is_rejected():
    X = np.arange(60, dtype=float).reshape(20, 3)
    model = CoverageCalibratedRandomADERH(
        n_estimators=2,
        n=6,
        calibration_max_points=0,
    )

    with pytest.raises(ValueError, match="calibration_max_points"):
        model.fit(X)
