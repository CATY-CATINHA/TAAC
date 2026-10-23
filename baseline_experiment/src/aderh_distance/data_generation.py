"""Synthetic data generation for the ADERH dimensionality experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

DEFAULT_DIMENSIONS = (2, 5, 10, 20, 50, 100, 200, 500)


@dataclass(frozen=True)
class SyntheticDataset:
    """One labeled synthetic dataset; labels are for splitting/evaluation only."""

    X: NDArray[np.float64]
    y: NDArray[np.int64]


def make_nested_datasets(
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    *,
    n_samples: int = 2_000,
    contamination: float = 0.05,
    random_state: int = 0,
) -> dict[int, SyntheticDataset]:
    """Generate datasets whose columns are nested across dimensionalities.

    The first two columns are drawn once. Normal observations follow
    ``N([0, 0], I)`` and anomalies follow ``N([4, 4], I)`` there. Every
    additional column is independent ``N(0, 1)`` noise for both classes.

    Noise is sampled feature-major before transposition. Consequently, calling
    this function with a different maximum dimensionality still produces the
    same prefix for a given seed; increasing dimensionality only appends noise.
    """

    requested = tuple(dict.fromkeys(int(dimension) for dimension in dimensions))
    if not requested:
        raise ValueError("at least one dimensionality is required")
    if min(requested) < 2:
        raise ValueError("all dimensionalities must be at least 2")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not 0.0 < contamination < 1.0:
        raise ValueError("contamination must lie strictly between 0 and 1")

    n_anomalies = int(round(n_samples * contamination))
    if not 0 < n_anomalies < n_samples:
        raise ValueError("contamination must produce both classes")

    rng = np.random.default_rng(random_state)
    labels = np.zeros(n_samples, dtype=np.int64)
    labels[-n_anomalies:] = 1
    labels = labels[rng.permutation(n_samples)]

    informative = rng.standard_normal((2, n_samples)).T
    informative[labels == 1] += 4.0

    max_dimension = max(requested)
    if max_dimension > 2:
        # Feature-major sampling makes prefixes invariant to max_dimension.
        irrelevant = rng.standard_normal((max_dimension - 2, n_samples)).T
        complete = np.concatenate((informative, irrelevant), axis=1)
    else:
        complete = informative

    return {
        dimension: SyntheticDataset(
            X=complete[:, :dimension].copy(),
            y=labels.copy(),
        )
        for dimension in requested
    }
