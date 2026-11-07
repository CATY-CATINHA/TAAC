import numpy as np

from pairing.common.base_aderh import BasePairingADERH


class HybridADERH(BasePairingADERH):
    """
    ADERH variant combining local and random pairing.

    alpha controls the proportion of anchors using local pairing.
    """

    def __init__(
        self,
        n_estimators=256,
        n=18,
        contamination=0.1,
        random_state=None,
        k_neighbors=5,
        alpha=0.5,
    ):
        super().__init__(
            n_estimators=n_estimators,
            n=n,
            contamination=contamination,
            random_state=random_state,
        )

        self.k_neighbors = k_neighbors
        self.alpha = alpha

    def _select_partners(self, X, anchor_idx, rnd):
        n_samples = X.shape[0]
        n_anchors = len(anchor_idx)

        if not 1 <= self.k_neighbors < n_samples:
            raise ValueError(
                "k_neighbors must be between 1 and n_samples - 1."
            )

        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        # Start with random pairing, following the original ADERH logic.
        perm = rnd.choice(
            n_anchors, n_anchors, replace=False
        )

        for j in range(n_anchors):
            if perm[j] == j:
                k = (j + 1) % n_anchors
                perm[j], perm[k] = perm[k], perm[j]

        partner_idx = anchor_idx[perm].copy()

        # Determine how many anchors will use local pairing.
        n_local = int(np.floor(
            self.alpha * n_anchors + 0.5
        ))

        if n_local == 0:
            return partner_idx

        # Randomly choose which anchors will use local pairing.
        local_positions = rnd.choice(
            n_anchors,
            size=n_local,
            replace=False
        )

        # Calculate distances from selected anchors to all points.
        distances = np.sum(
            (
                X[anchor_idx[local_positions], np.newaxis, :]
                - X[np.newaxis, :, :]
            ) ** 2,
            axis=2
        )

        # Exclude each anchor itself.
        distances[
            np.arange(n_local),
            anchor_idx[local_positions]
        ] = np.inf

        nearest_indices = np.argsort(
            distances, axis=1
        )[:, :self.k_neighbors]

        # Randomly select one of the k nearest neighbours.
        selected = rnd.randint(
            0, self.k_neighbors, size=n_local
        )

        partner_idx[local_positions] = nearest_indices[
            np.arange(n_local), selected
        ]

        return partner_idx