import numpy as np

from aderh import ADERH
from sklearn.utils import check_array, check_random_state
from scipy.optimize import linear_sum_assignment


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

    def _select_unique_partners(self, X, anchor_idx, rnd, local_positions):
        """
        Select partners while ensuring that:
        - every partner belongs to the sampled anchors;
        - no anchor is paired with itself;
        - every anchor is selected exactly once as a partner.

        Anchors in local_positions prefer their k nearest neighbours.
        Remaining anchors are paired randomly.
        """

        n_anchors = len(anchor_idx)
        local_positions = np.asarray(local_positions, dtype=int)

        # If no anchors use local pairing, preserve random derangement.
        if len(local_positions) == 0:
            perm = rnd.choice(n_anchors, n_anchors, replace=False)

            for j in range(n_anchors):
                if perm[j] == j:
                    k = (j + 1) % n_anchors
                    perm[j], perm[k] = perm[k], perm[j]

            return anchor_idx[perm]

        if not 1 <= self.k_neighbors < n_anchors:
            raise ValueError(
                "k_neighbors must be between 1 and n_anchors - 1."
            )

        # Calculate distances only between sampled anchors.
        anchors = X[anchor_idx]

        distances = np.sum(
            (anchors[:, np.newaxis, :] - anchors[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # An anchor cannot be its own partner.
        np.fill_diagonal(distances, np.inf)

        # Random costs for anchors using random pairing.
        random_costs = rnd.random((n_anchors, n_anchors))
        np.fill_diagonal(random_costs, np.inf)

        # Small random values break distance ties.
        tie_breakers = rnd.random((n_anchors, n_anchors)) * 1e-12

        k = self.k_neighbors

        while True:
            costs = random_costs.copy()

            # Restrict local anchors to their k nearest neighbours.
            for position in local_positions:
                nearest = np.argsort(distances[position])[:k]

                costs[position, :] = np.inf
                costs[position, nearest] = (
                    distances[position, nearest]
                    + tie_breakers[position, nearest]
                )

            try:
                rows, columns = linear_sum_assignment(costs)
                break

            except ValueError:
                # If no valid assignment exists, expand the neighbourhood.
                if k == n_anchors - 1:
                    raise RuntimeError(
                        "Could not construct a valid partner assignment."
                    )

                k += 1

        # Convert the assignment into original dataset indices.
        partner_positions = np.empty(n_anchors, dtype=int)
        partner_positions[rows] = columns

        return anchor_idx[partner_positions]

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

            # Recover partner positions within the sampled anchors.
            anchor_positions = {
                idx: position
                for position, idx in enumerate(anchor_idx)
            }

            partner_positions = np.array([
                anchor_positions[idx]
                for idx in partner_idx
            ])

            # Preserve the original ADERH radius calculation.
            sq_pair_dist = np.sum(
                (anchors - X[partner_positions]) ** 2,
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