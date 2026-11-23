"""ADERH ensemble whose estimators operate in random feature subspaces."""

from __future__ import annotations

import numpy as np
from aderh import ADERH
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_array, check_random_state


class RandomSubspaceADERH(ADERH):
    """Corrected random pairing in one random feature subset per estimator.

    ``max_features`` is an upper bound: datasets with fewer features use all
    available features. Pair centers and squared radii are built from actual
    sampled-anchor partners. When ``shuffle_radii`` is true, the per-estimator
    radius distribution is preserved but reassigned between center pairs.
    """

    def __init__(
        self,
        n_estimators=256,
        n=18,
        contamination=0.1,
        random_state=None,
        max_features=10,
        shuffle_radii=False,
    ):
        super().__init__(
            n_estimators=n_estimators,
            n=n,
            contamination=contamination,
            random_state=random_state,
        )
        self.max_features = max_features
        self.shuffle_radii = shuffle_radii

    def fit(self, X, y=None):
        """Fit corrected random-pair hyperspheres in random subspaces."""

        del y
        X = check_array(X, accept_sparse=False)
        n_samples, n_features = X.shape

        if self.n > n_samples:
            raise ValueError(
                f"n (={self.n}) must be <= n_samples (={n_samples})."
            )
        if self.n < 2:
            raise ValueError("n must be at least 2.")
        if not isinstance(self.max_features, (int, np.integer)):
            raise ValueError("max_features must be an integer")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if not isinstance(self.shuffle_radii, (bool, np.bool_)):
            raise ValueError("shuffle_radii must be boolean")

        n_anchors = self.n
        self.subspace_dim_ = min(int(self.max_features), n_features)
        self._feature_indices = np.empty(
            (self.n_estimators, self.subspace_dim_),
            dtype=int,
        )
        self._centers = np.empty(
            (
                self.n_estimators,
                2 * n_anchors,
                self.subspace_dim_,
            )
        )
        self._sq_radii = np.empty(
            (self.n_estimators, 2 * n_anchors)
        )

        random_state = check_random_state(self.random_state)
        max_int = np.iinfo(np.int32).max
        self._seeds = random_state.randint(
            max_int,
            size=self.n_estimators,
        )

        for estimator_index in range(self.n_estimators):
            rnd = check_random_state(self._seeds[estimator_index])

            anchor_idx = rnd.choice(
                n_samples,
                n_anchors,
                replace=False,
            )
            permutation = rnd.choice(
                n_anchors,
                n_anchors,
                replace=False,
            )
            for position in range(n_anchors):
                if permutation[position] == position:
                    swap = (position + 1) % n_anchors
                    permutation[position], permutation[swap] = (
                        permutation[swap],
                        permutation[position],
                    )

            feature_indices = np.sort(
                rnd.choice(
                    n_features,
                    self.subspace_dim_,
                    replace=False,
                )
            )
            self._feature_indices[estimator_index] = feature_indices

            anchors = X[anchor_idx][:, feature_indices]
            partners = anchors[permutation]
            squared_pair_distances = np.sum(
                (anchors - partners) ** 2,
                axis=1,
            )
            pair_radii = squared_pair_distances / 2
            if self.shuffle_radii:
                pair_radii = pair_radii[
                    rnd.permutation(n_anchors)
                ]

            self._centers[estimator_index] = np.concatenate(
                [anchors, partners],
                axis=0,
            )
            self._sq_radii[estimator_index] = np.concatenate(
                [pair_radii, pair_radii]
            )

        self.n_features_in_ = n_features
        self.decision_scores_ = self._raw_scores(X)
        self.threshold_ = np.percentile(
            self.decision_scores_,
            100 * (1 - self.contamination),
        )
        self.labels_ = (
            self.decision_scores_ > self.threshold_
        ).astype(int)
        return self

    def _raw_scores(self, X):
        """Reuse ADERH scoring within each estimator's feature subset."""

        X = check_array(X, accept_sparse=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same feature count used during fit"
            )

        per_estimator = np.ones((self.n_estimators, X.shape[0]))

        for estimator_index in range(self.n_estimators):
            squared_radii = np.where(
                self._sq_radii[estimator_index] != 0,
                self._sq_radii[estimator_index],
                1.0,
            )
            projected_X = X[:, self._feature_indices[estimator_index]]
            squared_distances = euclidean_distances(
                projected_X,
                self._centers[estimator_index],
                squared=True,
            )

            inside = np.where(
                squared_distances <= squared_radii,
                squared_distances,
                np.nan,
            )
            covered = np.where(~np.isnan(inside).all(axis=1))[0]
            if covered.size == 0:
                continue

            nearest = np.nanargmin(inside[covered], axis=1)
            pitch = inside[covered, nearest] / squared_radii[nearest]

            unique, counts = np.unique(nearest, return_counts=True)
            count_of = dict(zip(unique, counts))
            density = np.array(
                [count_of[index] for index in nearest],
                dtype=float,
            )
            density /= squared_radii[nearest]
            if density.size == 0:
                continue
            density /= density.max()

            per_estimator[estimator_index, covered] = (
                1.0 - density
            ) * pitch

        return np.mean(per_estimator, axis=0)
