import numpy as np
import pytest

from pairing.experiments.diagnostics import (
    hypersphere_geometry_diagnostics,
    neighbourhood_fidelity,
)
from pairing.experiments.experiments import run_experiment
from pairing.random_pairing.corrected_random_aderh import CorrectedRandomADERH


def test_neighbourhood_fidelity_is_one_for_identical_geometries():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))

    diagnostics = neighbourhood_fidelity(X, X.copy(), k=5)

    assert diagnostics["neighbour_fidelity_k"] == 5
    assert diagnostics["neighbour_fidelity_mean"] == pytest.approx(1.0)
    assert diagnostics["neighbour_fidelity_median"] == pytest.approx(1.0)


def test_irrelevant_noise_can_reduce_neighbourhood_fidelity():
    rng = np.random.default_rng(11)
    reference = np.linspace(0.0, 1.0, 80).reshape(-1, 1)
    noisy = np.concatenate(
        [reference, rng.normal(scale=20.0, size=(80, 20))],
        axis=1,
    )

    diagnostics = neighbourhood_fidelity(noisy, reference, k=5)

    assert 0.0 <= diagnostics["neighbour_fidelity_mean"] < 0.5
    assert diagnostics["neighbour_fidelity_min"] >= 0.0


def test_neighbourhood_fidelity_requires_aligned_rows():
    with pytest.raises(ValueError, match="same rows"):
        neighbourhood_fidelity(
            np.zeros((10, 2)),
            np.zeros((9, 2)),
            k=2,
        )


def test_corrected_random_geometry_has_aligned_pair_radius_scale():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(80, 5))
    model = CorrectedRandomADERH(
        n_estimators=6,
        n=10,
        random_state=3,
    ).fit(X)

    diagnostics = hypersphere_geometry_diagnostics(
        model,
        X,
        random_state=5,
        max_points=40,
        max_estimators=4,
    )

    assert diagnostics["geometry_point_sample_size"] == 40
    assert diagnostics["geometry_estimator_sample_size"] == 4
    assert diagnostics["radius_pair_scale_median"] == pytest.approx(1.0)
    assert diagnostics["pair_distance_median"] > 0.0
    assert diagnostics["radius_median"] > 0.0
    assert 0.0 <= diagnostics["sampled_train_estimator_coverage_mean"] <= 1.0
    assert 0.0 <= diagnostics["sampled_train_uncovered_fraction"] <= 1.0


def test_geometry_diagnostics_require_a_fitted_model():
    X = np.zeros((20, 2))

    with pytest.raises(ValueError, match="fitted"):
        hypersphere_geometry_diagnostics(CorrectedRandomADERH(), X)


def test_experiment_rejects_unknown_model_names(tmp_path):
    with pytest.raises(ValueError, match="unknown model"):
        run_experiment(
            dimensions=(2,),
            seeds=(0,),
            prefix="invalid",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            model_names=("not-a-model",),
        )
