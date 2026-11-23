import numpy as np

from pairing.common.base_aderh import BasePairingADERH


class CorrectedRandomADERH(BasePairingADERH):
    """Random ADERH control with actual pair distances and aligned radii."""

    def _select_partners(self, X, anchor_idx, rnd):
        return self._select_unique_partners(
            X,
            anchor_idx,
            rnd,
            np.array([], dtype=int),
        )
