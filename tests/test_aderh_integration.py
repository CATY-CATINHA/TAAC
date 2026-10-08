import numpy as np


def test_official_aderh_defaults_and_anomaly_score_direction() -> None:
    from aderh import ADERH

    rng = np.random.default_rng(41)
    X_train = rng.normal(loc=0.5, scale=0.1, size=(250, 2))
    model = ADERH(random_state=41).fit(X_train)

    params = model.get_params(deep=False)
    scores = model.decision_function(np.array([[0.5, 0.5], [20.0, 20.0]]))

    assert params["n_estimators"] == 256
    assert params["n"] == 18
    assert params["contamination"] == 0.1
    assert scores[1] > scores[0]

