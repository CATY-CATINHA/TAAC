
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
        strategy="rank"
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
        self.strategy = strategy

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

        if self.strategy not in ("rank", "continuous"):
            raise ValueError(
                "strategy must be 'rank' or 'continuous'."
            )

        if not 1 <= self.k_neighbors < n_anchors:
            raise ValueError(
                "k_neighbors must be between 1 and n_anchors - 1."
            )

        anchor_lids = self.lid_values_[anchor_idx]

        if self.strategy == "rank":

            # Original strategy: select a fixed proportion of anchors.
            n_local = int(np.floor(self.alpha * n_anchors + 0.5))

            if self.direction == "high_local":
                ranked_positions = np.argsort(-anchor_lids)
            else:
                ranked_positions = np.argsort(anchor_lids)

            local_positions = ranked_positions[:n_local]

        else:

            # Continuous strategy: assign a local-pairing probability
            # according to each anchor's relative LID.

            from scipy.stats import rankdata

            lid_ranks = rankdata(anchor_lids, method="average")

            lid_scores = (lid_ranks - 1) / (n_anchors - 1)

            if self.direction == "high_random":
                lid_scores = 1 - lid_scores

            probabilities = self.alpha * lid_scores

            # Each anchor independently decides whether to use local pairing.
            random_values = rnd.random(n_anchors)

            local_positions = np.flatnonzero(
                random_values < probabilities
            )

        return self._select_unique_partners(
            X,
            anchor_idx,
            rnd,
            local_positions,
        )
