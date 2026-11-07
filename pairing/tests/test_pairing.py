
import numpy as np
import pytest

from sklearn.base import clone
from aderh import ADERH

from pairing.local_pairing.local_aderh import LocalADERH
from pairing.hybrid_pairing.hybrid_aderh import HybridADERH


@pytest.fixture
def X():
    rng = np.random.RandomState(42)
    return rng.normal(size=(200, 5))


@pytest.mark.parametrize(
    "Model",
    [LocalADERH, HybridADERH]
)
def test_fit_produces_valid_scores(Model, X):
    model = Model(
        n_estimators=10,
        n=18,
        random_state=42
    )

    model.fit(X)

    assert model.decision_scores_.shape == (200,)
    assert np.all(np.isfinite(model.decision_scores_))
    assert model._centers.shape == (10, 36, 5)
    assert model._sq_radii.shape == (10, 36)


@pytest.mark.parametrize(
    "Model",
    [LocalADERH, HybridADERH]
)
def test_reproducibility(Model, X):
    model1 = Model(n_estimators=10, n=18, random_state=42)
    model2 = Model(n_estimators=10, n=18, random_state=42)

    model1.fit(X)
    model2.fit(X)

    assert np.array_equal(model1._centers, model2._centers)
    assert np.array_equal(model1._sq_radii, model2._sq_radii)
    assert np.array_equal(
        model1.decision_scores_,
        model2.decision_scores_
    )


@pytest.mark.parametrize(
    "Model",
    [LocalADERH, HybridADERH]
)
def test_sklearn_clone(Model):
    model = Model(random_state=42)
    cloned = clone(model)

    assert type(cloned) is Model


def test_local_pairing_selects_nearest_neighbor_when_k_is_one(X):
    model = LocalADERH(
        n_estimators=10,
        n=18,
        random_state=42,
        k_neighbors=1
    )

    model.fit(X)

    for estimator in range(model.n_estimators):
        anchors = model._centers[estimator, :model.n]
        partners = model._centers[estimator, model.n:]

        distances = np.sum(
            (anchors[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # Exclude each anchor itself.
        for i, anchor in enumerate(anchors):
            anchor_idx = np.where(
                np.all(X == anchor, axis=1)
            )[0][0]

            distances[i, anchor_idx] = np.inf

        nearest_distances = np.min(distances, axis=1)

        partner_distances = np.sum(
            (anchors - partners) ** 2,
            axis=1
        )

        assert np.allclose(
            partner_distances,
            nearest_distances
        )


def test_hybrid_alpha_zero_matches_original_centers(X):
    original = ADERH(
        n_estimators=10,
        n=18,
        random_state=42
    )

    hybrid = HybridADERH(
        n_estimators=10,
        n=18,
        random_state=42,
        alpha=0
    )

    original.fit(X)
    hybrid.fit(X)

    assert np.array_equal(
        original._centers,
        hybrid._centers
    )


def test_hybrid_alpha_one_uses_local_partners(X):
    model = HybridADERH(
        n_estimators=10,
        n=18,
        random_state=42,
        k_neighbors=1,
        alpha=1
    )

    model.fit(X)

    for estimator in range(model.n_estimators):
        anchors = model._centers[estimator, :model.n]
        partners = model._centers[estimator, model.n:]

        distances = np.sum(
            (anchors[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2,
            axis=2
        )

        for i, anchor in enumerate(anchors):
            anchor_idx = np.where(
                np.all(X == anchor, axis=1)
            )[0][0]

            distances[i, anchor_idx] = np.inf

        nearest_distances = np.min(distances, axis=1)
        partner_distances = np.sum(
            (anchors - partners) ** 2,
            axis=1
        )

        assert np.allclose(
            partner_distances,
            nearest_distances
        )


def test_hybrid_rejects_invalid_alpha(X):
    model = HybridADERH(alpha=1.5)

    with pytest.raises(ValueError):
        model.fit(X)


def test_local_rejects_invalid_k_neighbors(X):
    model = LocalADERH(k_neighbors=0)

    with pytest.raises(ValueError):
        model.fit(X)