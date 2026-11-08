
import numpy as np

from pairing.common.base_aderh import BasePairingADERH


class LocalADERH(BasePairingADERH):
    """
    ADERH variant where each anchor prefers a partner
    among its k nearest sampled anchors.
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
        n_anchors = len(anchor_idx)

        if not 1 <= self.k_neighbors < n_anchors:
            raise ValueError(
                "k_neighbors must be between 1 and n_anchors - 1."
            )

        # All anchors use local pairing.
        local_positions = np.arange(n_anchors)

        return self._select_unique_partners(
            X,
            anchor_idx,
            rnd,
            local_positions,
        )