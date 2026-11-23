"""Select one ADERH ensemble using the geometry of the training split."""

import numpy as np
from aderh import ADERH
from sklearn.utils.validation import check_array, check_is_fitted

from lid_analysis.lid import estimate_lid
from .random_subspace_aderh import RandomSubspaceADERH


class LIDGateADERH(ADERH):
    """Checkpoint 1. Compare median training LID with the ten-feature budget."""

    def fit(self, X, y=None):
        X = check_array(X)
        k, budget = 20, 10
        self.median_lid_ = np.nan
        self.n_valid_lid_ = 0
        if X.shape[1] <= budget:
            reason = "within_feature_budget"
        elif len(X) <= k:
            reason = "insufficient_neighbours"
        else:
            lid = estimate_lid(X, k=k, allow_degenerate=True)
            valid = lid[np.isfinite(lid)]
            self.n_valid_lid_ = len(valid)
            if len(valid):
                self.median_lid_ = float(np.median(valid))
            if 2 * len(valid) < len(X):
                reason = "insufficient_valid_lid"
            elif self.median_lid_ > budget:
                reason = "high_lid"
            else:
                reason = "low_lid"

        params = self.get_params(deep=False)
        self.route_reason_ = reason
        if reason == "high_lid":
            self.selected_model_ = "random_subspace_10_shuffled"
            detector = RandomSubspaceADERH(**params, max_features=budget, shuffle_radii=True)
        else:
            self.selected_model_ = "original"
            detector = ADERH(**params)
        self.detector_ = detector.fit(X)
        self.n_features_in_ = detector.n_features_in_
        self.decision_scores_ = detector.decision_scores_
        self.threshold_ = detector.threshold_
        self.labels_ = detector.labels_
        return self

    def decision_function(self, X):
        check_is_fitted(self, "detector_")
        return self.detector_.decision_function(X)
