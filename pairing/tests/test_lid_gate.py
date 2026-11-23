import numpy as np
import pytest
from aderh import ADERH
from sklearn.base import clone

from pairing.experiments.pre_subspace import CANDIDATE
from pairing.experiments.lid_gate import AMBIENT, GATE, evaluate
from pairing.subspace_pairing.lid_gate import LIDGateADERH
from pairing.subspace_pairing.random_subspace_aderh import RandomSubspaceADERH


@pytest.mark.parametrize("d,expected", [(5, "original"), (80, CANDIDATE)])
def test_gate_matches_independently_fitted_branch(d, expected):
    rng = np.random.default_rng(14)
    X, test = rng.normal(size=(200, d)), rng.normal(size=(41, d))
    params = dict(n_estimators=5, n=8, random_state=74)
    gate = clone(LIDGateADERH(**params)).fit(X, y=np.arange(len(X)))
    model = (ADERH(**params) if expected == "original" else
             RandomSubspaceADERH(**params, max_features=10, shuffle_radii=True)).fit(X)
    assert gate.selected_model_ == expected
    np.testing.assert_array_equal(gate.decision_function(test), model.decision_function(test))
    np.testing.assert_array_equal(gate.predict(test), model.predict(test))
    np.testing.assert_array_equal(gate.labels_, model.labels_)


@pytest.mark.parametrize("values,expected", [
    ([10.0]*40, "low_lid"), ([11.0]*20 + [np.nan]*20, "high_lid"),
    ([11.0]*19 + [np.nan]*21, "insufficient_valid_lid"),
])
def test_lid_boundary_and_minimum_coverage(monkeypatch, values, expected):
    monkeypatch.setattr("pairing.subspace_pairing.lid_gate.estimate_lid",
                        lambda X, **kwargs: np.array(values))
    X = np.random.default_rng(3).normal(size=(40, 12))
    gate = LIDGateADERH(n_estimators=2, n=8, random_state=0).fit(X)
    assert gate.route_reason_ == expected


def test_duplicate_training_points_and_too_few_neighbours():
    gate = LIDGateADERH(n_estimators=2, n=8, random_state=0)
    gate.fit(np.ones((40, 12)))
    assert gate.route_reason_ == "insufficient_valid_lid"
    assert gate.n_valid_lid_ == 0
    gate.fit(np.ones((20, 12)))
    assert gate.route_reason_ == "insufficient_neighbours"


def test_high_ambient_low_intrinsic_uses_different_controls():
    X = np.tile(np.arange(80, dtype=float)[:, None], (1, 12))
    y = np.tile([0, 1], 40)
    rows = {row["model"]: row for row in
            evaluate("line", X, y, np.arange(60), np.arange(60, 80), 0, [1])}
    assert rows[GATE]["selected_model"] == "original"
    assert rows[AMBIENT]["selected_model"] == CANDIDATE
    for metric in ("roc_auc", "average_precision", "uncovered_fraction"):
        assert rows[GATE][metric] == rows["original"][metric]
        assert rows[AMBIENT][metric] == rows[CANDIDATE][metric]
