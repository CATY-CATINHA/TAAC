
import numpy as np

from pairing.common.base_aderh import BasePairingADERH
from lid_analysis.lid import estimate_lid


class LIDAdaptiveADERH(BasePairingADERH):
    """
    ADERH variant that uses Local Intrinsic Dimensionality (LID)
    to select which anchors use local pairing.

    direction:
        - "high_local": higher LID -> local pairing
        - "high_random": higher LID -> random pairing
    """

    def __init__(
        self,
        n_estimators=256,
        n=18,
        contamination=0.1,
        random_state=None,
        k_neighbors=5,
        alpha=0.5,
        lid_k=20,
        direction="high_local",
    ):
        super().__init__(
            n_estimators=n_estimators,
            n=n,
            contamination=contamination,
            random_state=random_state,
        )

        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.lid_k = lid_k
        self.direction = direction

    def fit(self, X, y=None):
        """
        Estimate LID once for the training data,
        then reuse the original BasePairingADERH fit.
        """

        # Estimate LID for every training observation.
        self.lid_values_ = estimate_lid(X, k=self.lid_k)

        return super().fit(X, y)

    def _select_partners(self, X, anchor_idx, rnd):
        n_anchors = len(anchor_idx)

        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        if self.direction not in ("high_local", "high_random"):
            raise ValueError(
                "direction must be 'high_local' or 'high_random'."
            )

        if not 1 <= self.k_neighbors < n_anchors:
            raise ValueError(
                "k_neighbors must be between 1 and n_anchors - 1."
            )

        n_local = int(np.floor(self.alpha * n_anchors + 0.5))

        # LID values corresponding to the sampled anchors.
        anchor_lids = self.lid_values_[anchor_idx]

        # Rank anchors according to their LID.
        if self.direction == "high_local":
            # Highest LID values are selected for local pairing.
            ranked_positions = np.argsort(-anchor_lids)
        else:
            # Lowest LID values are selected for local pairing.
            ranked_positions = np.argsort(anchor_lids)

        # Select exactly n_local anchors.
        local_positions = ranked_positions[:n_local]

        return self._select_unique_partners(
            X,
            anchor_idx,
            rnd,
            local_positions,
        )
