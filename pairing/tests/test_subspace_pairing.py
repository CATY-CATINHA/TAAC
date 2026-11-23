import numpy as np
import pytest
from sklearn.base import clone

from pairing.experiments.diagnostics import (
    hypersphere_geometry_diagnostics,
)
from pairing.random_pairing.corrected_random_aderh import CorrectedRandomADERH
from pairing.subspace_pairing.random_subspace_aderh import (
    RandomSubspaceADERH,
)


def test_low_dimension_subspace_matches_corrected_random():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(100, 2))
    corrected = CorrectedRandomADERH(
        n_estimators=8,
        n=12,
        random_state=7,
    ).fit(X)
    subspace = RandomSubspaceADERH(
        n_estimators=8,
        n=12,
        random_state=7,
        max_features=10,
    ).fit(X)

    assert subspace.subspace_dim_ == 2
    assert np.array_equal(
        subspace._feature_indices,
        np.tile(np.arange(2), (8, 1)),
    )
    assert np.array_equal(subspace._centers, corrected._centers)
    assert np.array_equal(subspace._sq_radii, corrected._sq_radii)
    assert np.allclose(
        subspace.decision_scores_,
        corrected.decision_scores_,
    )


def test_random_subspace_uses_valid_feature_sets_and_scores():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(120, 25))
    model = RandomSubspaceADERH(
        n_estimators=10,
        n=12,
        random_state=8,
        max_features=7,
    ).fit(X)

    assert model._feature_indices.shape == (10, 7)
    assert model._centers.shape == (10, 24, 7)
    assert model._sq_radii.shape == (10, 24)
    assert np.all(model._feature_indices >= 0)
    assert np.all(model._feature_indices < 25)
    assert all(
        np.unique(indices).size == 7
        for indices in model._feature_indices
    )
    assert np.all(np.isfinite(model.decision_scores_))
    assert np.all(np.isfinite(model.decision_function(X[:20])))


def test_subspace_radius_shuffle_preserves_geometry_distribution():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(100, 20))
    aligned = RandomSubspaceADERH(
        n_estimators=8,
        n=12,
        random_state=9,
        max_features=6,
        shuffle_radii=False,
    ).fit(X)
    shuffled = RandomSubspaceADERH(
        n_estimators=8,
        n=12,
        random_state=9,
        max_features=6,
        shuffle_radii=True,
    ).fit(X)

    assert np.array_equal(aligned._feature_indices, shuffled._feature_indices)
    assert np.array_equal(aligned._centers, shuffled._centers)
    assert np.allclose(
        np.sort(aligned._sq_radii[:, : aligned.n], axis=1),
        np.sort(shuffled._sq_radii[:, : shuffled.n], axis=1),
    )
    assert not np.array_equal(aligned._sq_radii, shuffled._sq_radii)


def test_subspace_geometry_diagnostics_use_estimator_features():
    rng = np.random.default_rng(33)
    X = rng.normal(size=(100, 15))
    model = RandomSubspaceADERH(
        n_estimators=6,
        n=10,
        random_state=10,
        max_features=5,
    ).fit(X)

    diagnostics = hypersphere_geometry_diagnostics(
        model,
        X,
        max_points=40,
        max_estimators=4,
    )

    assert diagnostics["geometry_point_sample_size"] == 40
    assert diagnostics["geometry_estimator_sample_size"] == 4
    assert diagnostics["effective_feature_count"] == 5
    assert 0.0 <= diagnostics["informative_feature_any_fraction"] <= 1.0
    assert 0.0 <= diagnostics["informative_feature_both_fraction"] <= 1.0
    assert 0.0 <= diagnostics["informative_feature_count_mean"] <= 2.0
    assert diagnostics["radius_pair_scale_median"] == pytest.approx(1.0)


def test_random_subspace_is_reproducible_and_cloneable():
    rng = np.random.default_rng(34)
    X = rng.normal(size=(100, 12))
    first = RandomSubspaceADERH(
        n_estimators=6,
        n=10,
        random_state=11,
        max_features=4,
        shuffle_radii=True,
    ).fit(X)
    second = clone(first).fit(X)

    assert np.array_equal(first._feature_indices, second._feature_indices)
    assert np.array_equal(first._centers, second._centers)
    assert np.array_equal(first._sq_radii, second._sq_radii)
    assert np.array_equal(first.decision_scores_, second.decision_scores_)


@pytest.mark.parametrize("max_features", [0, -1, 1.5])
def test_random_subspace_rejects_invalid_max_features(max_features):
    X = np.arange(60, dtype=float).reshape(20, 3)
    model = RandomSubspaceADERH(
        n_estimators=2,
        n=6,
        max_features=max_features,
    )

    with pytest.raises(ValueError, match="max_features"):
        model.fit(X)
