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
        n_anchors = len(anchor_idx)

        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        n_local = int(np.floor(self.alpha * n_anchors + 0.5))

        # Randomly select which anchors use local pairing.
        if n_local > 0:
            local_positions = rnd.choice(
                n_anchors,
                size=n_local,
                replace=False
            )
        else:
            local_positions = np.array([], dtype=int)

        return self._select_unique_partners(
            X,
            anchor_idx,
            rnd,
            local_positions
        )