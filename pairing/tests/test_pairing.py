
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


def test_local_pairing_uses_unique_sampled_partners(X):
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

        # Every partner must belong to the sampled anchors.
        for partner in partners:
            assert any(np.array_equal(partner, anchor) for anchor in anchors)

        # No anchor can be paired with itself.
        for anchor, partner in zip(anchors, partners):
            assert not np.array_equal(anchor, partner)

        # Every anchor must be selected exactly once as a partner.
        for anchor in anchors:
            occurrences = sum(
                np.array_equal(anchor, partner)
                for partner in partners
            )
            assert occurrences == 1


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


def test_hybrid_alpha_one_uses_unique_sampled_partners(X):
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

        # Every partner must belong to the sampled anchors.
        for partner in partners:
            assert any(
                np.array_equal(partner, anchor)
                for anchor in anchors
            )

        # No anchor can be paired with itself.
        for anchor, partner in zip(anchors, partners):
            assert not np.array_equal(anchor, partner)

        # Every anchor must appear exactly once as a partner.
        for anchor in anchors:
            occurrences = sum(
                np.array_equal(anchor, partner)
                for partner in partners
            )
            assert occurrences == 1

def test_hybrid_alpha_half_selects_half_local_anchors(X, monkeypatch):
    model = HybridADERH(
        n_estimators=10,
        n=18,
        random_state=42,
        k_neighbors=1,
        alpha=0.5
    )

    captured = {}
    original_method = model._select_unique_partners

    def capture_local_positions(X, anchor_idx, rnd, local_positions):
        captured["local_positions"] = np.asarray(local_positions)
        return original_method(X, anchor_idx, rnd, local_positions)

    monkeypatch.setattr(
        model,
        "_select_unique_partners",
        capture_local_positions
    )

    anchor_idx = np.arange(model.n)
    rnd = np.random.RandomState(42)

    model._select_partners(X, anchor_idx, rnd)

    local_positions = captured["local_positions"]

    assert len(local_positions) == 9
    assert len(np.unique(local_positions)) == 9
    assert np.all(local_positions >= 0)
    assert np.all(local_positions < model.n)

def test_hybrid_rejects_invalid_alpha(X):
    model = HybridADERH(alpha=1.5)

    with pytest.raises(ValueError):
        model.fit(X)


def test_local_rejects_invalid_k_neighbors(X):
    model = LocalADERH(k_neighbors=0)

    with pytest.raises(ValueError):
        model.fit(X)