"""Training-only hypersphere coverage utilities.

Coverage calibration operates on geometry alone. It does not use anomaly
labels, decision scores or test observations.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_array


def sample_coverage_indices(
    *,
    n_points: int,
    n_estimators: int,
    random_state: int,
    max_points: int,
    max_estimators: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select reproducible point and estimator subsets for calibration."""

    if n_points < 1 or n_estimators < 1:
        raise ValueError("n_points and n_estimators must be positive")
    if max_points < 1 or max_estimators < 1:
        raise ValueError("sample limits must be positive")

    rng = np.random.default_rng(random_state)
    point_indices = rng.choice(
        n_points,
        size=min(n_points, max_points),
        replace=False,
    )
    estimator_indices = rng.choice(
        n_estimators,
        size=min(n_estimators, max_estimators),
        replace=False,
    )
    return point_indices, estimator_indices


def coverage_scale_thresholds(
    X: np.ndarray,
    centers: np.ndarray,
    squared_radii: np.ndarray,
    *,
    point_indices: np.ndarray,
    estimator_indices: np.ndarray,
) -> np.ndarray:
    """Return the squared-radius scale needed to cover each sampled case.

    For every sampled point/estimator combination, the returned value is the
    minimum ``gamma`` for which at least one sphere covers the point when all
    squared radii are multiplied by ``gamma``.
    """

    X = check_array(X, ensure_min_samples=1, ensure_2d=True)
    centers = np.asarray(centers, dtype=float)
    squared_radii = np.asarray(squared_radii, dtype=float)
    point_indices = np.asarray(point_indices, dtype=int)
    estimator_indices = np.asarray(estimator_indices, dtype=int)

    if centers.ndim != 3 or squared_radii.ndim != 2:
        raise ValueError("centers and squared_radii have invalid shapes")
    if centers.shape[:2] != squared_radii.shape:
        raise ValueError("centers and squared_radii must be aligned")
    if centers.shape[2] != X.shape[1]:
        raise ValueError("centers and X must have the same feature count")
    if np.any(squared_radii < 0) or not np.all(np.isfinite(squared_radii)):
        raise ValueError("squared_radii must be finite and non-negative")
    if point_indices.size == 0 or estimator_indices.size == 0:
        raise ValueError("coverage samples must not be empty")
    if np.any((point_indices < 0) | (point_indices >= X.shape[0])):
        raise ValueError("point_indices are out of range")
    if np.any(
        (estimator_indices < 0)
        | (estimator_indices >= centers.shape[0])
    ):
        raise ValueError("estimator_indices are out of range")

    sampled_points = X[point_indices]
    thresholds = np.empty(
        (estimator_indices.size, point_indices.size),
        dtype=float,
    )

    for output_index, estimator_index in enumerate(estimator_indices):
        estimator_centers = centers[estimator_index]
        differences = (
            sampled_points[:, np.newaxis, :]
            - estimator_centers[np.newaxis, :, :]
        )
        squared_distances = np.einsum(
            "pcd,pcd->pc",
            differences,
            differences,
        )
        estimator_radii = squared_radii[estimator_index][np.newaxis, :]
        ratios = np.divide(
            squared_distances,
            estimator_radii,
            out=np.full_like(squared_distances, np.inf),
            where=estimator_radii > 0,
        )
        ratios[(squared_distances == 0) & (estimator_radii == 0)] = 0.0
        thresholds[output_index] = np.min(ratios, axis=1)

    return thresholds.reshape(-1)


def coverage_at_scale(thresholds: np.ndarray, gamma: float) -> float:
    """Calculate mean sampled estimator coverage at squared-radius scale."""

    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")
    if gamma < 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and non-negative")
    return float(np.mean(thresholds <= gamma))


def scale_for_target_coverage(
    thresholds: np.ndarray,
    target_coverage: float,
) -> float:
    """Select the smallest empirical scale attaining target coverage."""

    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")
    if not 0 <= target_coverage <= 1:
        raise ValueError("target_coverage must lie in [0, 1]")

    finite = thresholds[np.isfinite(thresholds)]
    if finite.size == 0:
        raise ValueError("coverage cannot be achieved with finite radii")
    if target_coverage == 0:
        return 0.0
    if target_coverage > finite.size / thresholds.size:
        raise ValueError("target coverage requires an infinite radius scale")

    ordered = np.sort(finite)
    required_count = int(np.ceil(target_coverage * thresholds.size))
    required_count = max(1, required_count)
    return float(ordered[required_count - 1])
