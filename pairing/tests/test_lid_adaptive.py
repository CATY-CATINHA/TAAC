import numpy as np
import pytest
from sklearn.datasets import make_blobs

from pairing.lid_adaptive_pairing.lid_adaptive_aderh import LIDAdaptiveADERH


@pytest.fixture
def X():
    data, _ = make_blobs(
        n_samples=200,
        n_features=5,
        centers=3,
        random_state=42,
    )
    return data


@pytest.mark.parametrize("direction", ["high_local", "high_random"])
def test_fit_produces_valid_lid_and_scores(X, direction):
    model = LIDAdaptiveADERH(
        n_estimators=10,
        n=18,
        contamination=0.1,
        random_state=42,
        k_neighbors=5,
        alpha=0.5,
        lid_k=20,
        direction=direction,
    )

    model.fit(X)

    assert model.lid_values_.shape == (200,)
    assert model.decision_scores_.shape == (200,)
    assert np.all(np.isfinite(model.lid_values_))
    assert np.all(np.isfinite(model.decision_scores_))
    assert model._centers.shape == (10, 36, 5)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_continuous_probabilities_have_alpha_mean(alpha):
    model = LIDAdaptiveADERH(alpha=alpha, strategy="continuous")
    scores = np.linspace(0.0, 1.0, model.n)

    probabilities = model._continuous_probabilities(scores)

    assert np.all((0 <= probabilities) & (probabilities <= 1))
    assert np.mean(probabilities) == pytest.approx(alpha)
    assert np.all(np.diff(probabilities) >= 0)


def test_continuous_directions_have_same_expected_local_fraction(monkeypatch):
    lids = np.arange(18, dtype=float)
    X = np.arange(18, dtype=float).reshape(-1, 1)
    captured = {}

    class ConstantRandom:
        @staticmethod
        def random(size):
            return np.full(size, 0.49)

    def capture(X, anchor_idx, rnd, local_positions):
        captured[model.direction] = np.asarray(local_positions)
        return np.roll(anchor_idx, 1)

    for direction in ("high_local", "high_random"):
        model = LIDAdaptiveADERH(
            n=18,
            alpha=0.5,
            direction=direction,
            strategy="continuous",
        )
        model.lid_values_ = lids
        monkeypatch.setattr(model, "_select_unique_partners", capture)
        model._select_partners(X, np.arange(18), ConstantRandom())

    assert len(captured["high_local"]) == len(captured["high_random"])


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"alpha": -0.1}, "alpha"),
        ({"direction": "invalid"}, "direction"),
        ({"strategy": "invalid"}, "strategy"),
        ({"k_neighbors": 0}, "k_neighbors"),
    ],
)
def test_invalid_configuration_is_rejected(X, parameters, message):
    model = LIDAdaptiveADERH(n_estimators=1, **parameters)

    with pytest.raises(ValueError, match=message):
        model.fit(X)
