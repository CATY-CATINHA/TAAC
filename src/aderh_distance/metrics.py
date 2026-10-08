"""Geometric metrics kept separate from detector performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array


@dataclass(frozen=True)
class RelativeContrastResult:
    """Relative contrast and the sampling counts used to estimate it."""

    value: float
    mean_random_distance: float
    mean_nearest_neighbor_distance: float
    reference_sample_size: int
    random_pair_count: int


def relative_contrast(
    X_normal: ArrayLike,
    *,
    random_state: int = 0,
    reference_sample_size: int = 512,
    random_pair_count: int = 4_096,
) -> RelativeContrastResult:
    """Estimate relative contrast on normal reference observations.

    The operational definition is

    ``mean(distance between random distinct reference pairs) /``
    ``mean(nearest-neighbor distance within the reference sample)``.

    A seeded subset of at most ``reference_sample_size`` rows is shared by both
    terms. The numerator uses ``random_pair_count`` uniformly sampled ordered
    pairs with self-pairs excluded. The denominator uses a nearest-neighbor
    index and discards each query row's self-match. No full pairwise distance
    matrix is materialized.
    """

    X = check_array(X_normal, ensure_min_samples=2, ensure_2d=True)
    if reference_sample_size < 2:
        raise ValueError("reference_sample_size must be at least 2")
    if random_pair_count < 1:
        raise ValueError("random_pair_count must be positive")

    rng = np.random.default_rng(random_state)
    sample_size = min(reference_sample_size, X.shape[0])
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    reference = X[sample_indices]

    left = rng.integers(0, sample_size, size=random_pair_count)
    # Adding an offset in [1, sample_size) gives uniform, distinct partners.
    offsets = rng.integers(1, sample_size, size=random_pair_count)
    right = (left + offsets) % sample_size
    random_distances = np.linalg.norm(reference[left] - reference[right], axis=1)

    neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean")
    neighbors.fit(reference)
    distances, _ = neighbors.kneighbors(reference, return_distance=True)
    nearest_distances = distances[:, 1]

    mean_random = float(np.mean(random_distances))
    mean_nearest = float(np.mean(nearest_distances))
    if mean_nearest == 0.0:
        value = float("inf")
    else:
        value = mean_random / mean_nearest

    return RelativeContrastResult(
        value=float(value),
        mean_random_distance=mean_random,
        mean_nearest_neighbor_distance=mean_nearest,
        reference_sample_size=sample_size,
        random_pair_count=random_pair_count,
    )
