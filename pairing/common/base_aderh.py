import numpy as np

from aderh import ADERH
from sklearn.utils import check_array, check_random_state


class BasePairingADERH(ADERH):
    """
    Base class for experimental ADERH pairing strategies.

    Reuses the original ADERH scoring mechanism and changes
    only the selection of pair partners.
    """

    def _select_partners(self, X, anchor_idx, rnd):
        """
        Select partners for the sampled anchors.

        Must be implemented by each pairing strategy.
        """
        raise NotImplementedError

    def fit(self, X, y=None):
        """Fit ADERH using the selected pairing strategy."""

        X = check_array(X, accept_sparse=False)
        n_samples, n_features = X.shape

        if self.n > n_samples:
            raise ValueError(
                f"n (={self.n}) must be <= n_samples (={n_samples})."
            )

        if self.n < 2:
            raise ValueError("n must be at least 2.")

        n_anchors = self.n

        self._centers = np.empty(
            (self.n_estimators, 2 * n_anchors, n_features)
        )
        self._sq_radii = np.empty(
            (self.n_estimators, 2 * n_anchors)
        )

        random_state = check_random_state(self.random_state)
        max_int = np.iinfo(np.int32).max

        self._seeds = random_state.randint(
            max_int,
            size=self.n_estimators
        )

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])

            anchor_idx = rnd.choice(
                n_samples,
                n_anchors,
                replace=False
            )

            anchors = X[anchor_idx]

            partner_idx = self._select_partners(
                X, anchor_idx, rnd
            )
            partners = X[partner_idx]

            sq_pair_dist = np.sum(
                (anchors - partners) ** 2,
                axis=1
            )

            self._centers[i] = np.concatenate(
                [anchors, partners],
                axis=0
            )

            self._sq_radii[i] = np.repeat(
                sq_pair_dist.reshape(-1, 1) / 2,
                2,
                axis=1
            ).reshape(-1)

        self.n_features_in_ = n_features

        # Reuse the original ADERH scoring mechanism.
        self.decision_scores_ = self._raw_scores(X)

        self.threshold_ = np.percentile(
            self.decision_scores_,
            100 * (1 - self.contamination)
        )

        self.labels_ = (
            self.decision_scores_ > self.threshold_
        ).astype(int)

        return self