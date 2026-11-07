import numpy as np

from pairing.common.base_aderh import BasePairingADERH


class LocalADERH(BasePairingADERH):
    """
    ADERH variant where each anchor is paired with a randomly
    selected point among its k nearest neighbours.
    """

    def __init__(
        self,
        n_estimators=256,
        n=18,
        contamination=0.1,
        random_state=None,
        k_neighbors=5,
    ):
        super().__init__(
            n_estimators=n_estimators,
            n=n,
            contamination=contamination,
            random_state=random_state,
        )

        self.k_neighbors = k_neighbors

    def _select_partners(self, X, anchor_idx, rnd):
        n_samples = X.shape[0]
        n_anchors = len(anchor_idx)

        if not 1 <= self.k_neighbors < n_samples:
            raise ValueError(
                "k_neighbors must be between 1 and n_samples - 1."
            )

        # Calculate squared distances from each anchor to all points.
        distances = np.sum(
            (X[anchor_idx, np.newaxis, :] - X[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # Exclude each anchor itself.
        distances[
            np.arange(n_anchors),
            anchor_idx
        ] = np.inf

        # Identify the k nearest neighbours.
        nearest_indices = np.argsort(
            distances, axis=1
        )[:, :self.k_neighbors]

        # Randomly select one of the k nearest neighbours.
        selected = rnd.randint(
            0, self.k_neighbors, size=n_anchors
        )

        return nearest_indices[
            np.arange(n_anchors), selected
        ]