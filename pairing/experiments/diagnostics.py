"""Geometry diagnostics used to design the second experimental phase.

The functions in this module do not tune a detector. They expose whether a
pairing policy changes neighbourhood fidelity, radius scale and hypersphere
coverage in the way intended by the experimental hypothesis.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array


def _nearest_neighbour_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Return ``k`` neighbours per row, excluding the row itself."""

    n_samples = X.shape[0]
    if not 1 <= k < n_samples:
        raise ValueError("k must be between 1 and n_samples - 1")

    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="brute",
    )
    model.fit(X)
    candidates = model.kneighbors(X, return_distance=False)

    neighbours = np.empty((n_samples, k), dtype=int)
    for row_index, row_candidates in enumerate(candidates):
        without_self = row_candidates[row_candidates != row_index]
        if without_self.size < k:
            raise RuntimeError("Could not exclude the self-neighbour")
        neighbours[row_index] = without_self[:k]

    return neighbours


def neighbourhood_fidelity(
    X: np.ndarray,
    reference: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, float | int]:
    """Compare neighbours in ``X`` with neighbours in a reference geometry.

    Both arrays must contain the same observations in the same row order. In
    the controlled synthetic experiment, ``reference`` is the two-dimensional
    informative prefix and ``X`` is the complete ambient representation.

    The overlap is a diagnostic that uses known synthetic structure. It is not
    available to the detector and must not be used as a real-data tuning
    signal.
    """

    X = check_array(X, ensure_min_samples=2, ensure_2d=True)
    reference = check_array(
        reference,
        ensure_min_samples=2,
        ensure_2d=True,
    )

    if X.shape[0] != reference.shape[0]:
        raise ValueError("X and reference must contain the same rows")

    ambient_neighbours = _nearest_neighbour_indices(X, k)
    reference_neighbours = _nearest_neighbour_indices(reference, k)

    overlap = np.fromiter(
        (
            np.intersect1d(
                ambient_neighbours[row],
                reference_neighbours[row],
                assume_unique=True,
            ).size
            / k
            for row in range(X.shape[0])
        ),
        dtype=float,
        count=X.shape[0],
    )

    return {
        "neighbour_fidelity_k": int(k),
        "neighbour_fidelity_mean": float(np.mean(overlap)),
        "neighbour_fidelity_std": float(np.std(overlap, ddof=0)),
        "neighbour_fidelity_q10": float(np.quantile(overlap, 0.10)),
        "neighbour_fidelity_median": float(np.median(overlap)),
        "neighbour_fidelity_min": float(np.min(overlap)),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a Pearson correlation, or NaN for constant inputs."""

    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def hypersphere_geometry_diagnostics(
    model,
    X: np.ndarray,
    *,
    random_state: int = 0,
    max_points: int = 512,
    max_estimators: int = 64,
) -> dict[str, float | int]:
    """Summarise fitted pairs, radii and sampled training coverage.

    Radius and pair-distance summaries use the complete fitted ensemble.
    Coverage is estimated on a seeded sample of observations and estimators to
    keep the diagnostic inexpensive enough for development experiments. The
    returned sample sizes make that approximation explicit.
    """

    X = check_array(X, ensure_min_samples=2, ensure_2d=True)

    if max_points < 1 or max_estimators < 1:
        raise ValueError("max_points and max_estimators must be positive")

    try:
        centers = np.asarray(model._centers, dtype=float)
        squared_radii = np.asarray(model._sq_radii, dtype=float)
    except AttributeError as error:
        raise ValueError("model must be fitted before diagnostics") from error

    if centers.ndim != 3 or squared_radii.ndim != 2:
        raise ValueError("invalid fitted center or radius shape")
    if centers.shape[:2] != squared_radii.shape:
        raise ValueError("centers and radii must be aligned")

    feature_indices = getattr(model, "_feature_indices", None)
    if feature_indices is None:
        if centers.shape[2] != X.shape[1]:
            raise ValueError("model and X must have the same feature count")
    else:
        feature_indices = np.asarray(feature_indices, dtype=int)
        if feature_indices.shape != (
            centers.shape[0],
            centers.shape[2],
        ):
            raise ValueError("invalid per-estimator feature-index shape")
        if np.any((feature_indices < 0) | (feature_indices >= X.shape[1])):
            raise ValueError("feature indices are out of range")
    if centers.shape[1] % 2 != 0:
        raise ValueError("the center count must contain two blocks")
    if np.any(squared_radii < 0) or not np.all(np.isfinite(squared_radii)):
        raise ValueError("squared radii must be finite and non-negative")

    n_estimators, n_centers, _ = centers.shape
    n_pairs = n_centers // 2

    anchors = centers[:, :n_pairs]
    partners = centers[:, n_pairs:]
    pair_distances = np.linalg.norm(anchors - partners, axis=2)
    radii = np.sqrt(squared_radii)

    first_block_radii = radii[:, :n_pairs]
    expected_pair_radii = pair_distances / np.sqrt(2.0)
    valid_scale = expected_pair_radii > np.finfo(float).eps
    radius_pair_scale = np.divide(
        first_block_radii,
        expected_pair_radii,
        out=np.full_like(first_block_radii, np.nan),
        where=valid_scale,
    )

    if feature_indices is None:
        centroid = np.mean(X, axis=0)
        center_eccentricity = np.linalg.norm(
            centers - centroid[np.newaxis, np.newaxis, :],
            axis=2,
        )
    else:
        center_eccentricity = np.empty(
            (n_estimators, n_centers),
            dtype=float,
        )
        for estimator_index in range(n_estimators):
            projected_centroid = np.mean(
                X[:, feature_indices[estimator_index]],
                axis=0,
            )
            center_eccentricity[estimator_index] = np.linalg.norm(
                centers[estimator_index] - projected_centroid,
                axis=1,
            )

    rng = np.random.default_rng(random_state)
    point_count = min(max_points, X.shape[0])
    estimator_count = min(max_estimators, n_estimators)
    point_indices = rng.choice(X.shape[0], size=point_count, replace=False)
    estimator_indices = rng.choice(
        n_estimators,
        size=estimator_count,
        replace=False,
    )

    sampled_points = X[point_indices]
    covering_estimators = np.zeros(point_count, dtype=int)
    coverage_by_estimator = np.empty(estimator_count, dtype=float)

    for output_index, estimator_index in enumerate(estimator_indices):
        estimator_centers = centers[estimator_index]
        if feature_indices is None:
            estimator_points = sampled_points
        else:
            estimator_points = sampled_points[
                :, feature_indices[estimator_index]
            ]
        differences = (
            estimator_points[:, np.newaxis, :]
            - estimator_centers[np.newaxis, :, :]
        )
        point_center_distances = np.einsum(
            "pcd,pcd->pc",
            differences,
            differences,
        )
        covered = np.any(
            point_center_distances
            <= squared_radii[estimator_index][np.newaxis, :],
            axis=1,
        )
        covering_estimators += covered
        coverage_by_estimator[output_index] = np.mean(covered)

    finite_scales = radius_pair_scale[np.isfinite(radius_pair_scale)]
    radius_pair_scale_median = (
        float(np.median(finite_scales))
        if finite_scales.size
        else float("nan")
    )

    if feature_indices is None:
        effective_feature_count = X.shape[1]
        informative_feature_any_fraction = 1.0
        informative_feature_both_fraction = 1.0
        informative_feature_count_mean = float(min(2, X.shape[1]))
    else:
        informative_counts = np.sum(feature_indices < 2, axis=1)
        effective_feature_count = feature_indices.shape[1]
        informative_feature_any_fraction = float(
            np.mean(informative_counts >= 1)
        )
        informative_feature_both_fraction = float(
            np.mean(informative_counts == 2)
        )
        informative_feature_count_mean = float(
            np.mean(informative_counts)
        )

    return {
        "geometry_point_sample_size": int(point_count),
        "geometry_estimator_sample_size": int(estimator_count),
        "effective_feature_count": int(effective_feature_count),
        "informative_feature_any_fraction": (
            informative_feature_any_fraction
        ),
        "informative_feature_both_fraction": (
            informative_feature_both_fraction
        ),
        "informative_feature_count_mean": informative_feature_count_mean,
        "pair_distance_mean": float(np.mean(pair_distances)),
        "pair_distance_median": float(np.median(pair_distances)),
        "pair_distance_q10": float(np.quantile(pair_distances, 0.10)),
        "pair_distance_q90": float(np.quantile(pair_distances, 0.90)),
        "radius_mean": float(np.mean(radii)),
        "radius_median": float(np.median(radii)),
        "radius_q10": float(np.quantile(radii, 0.10)),
        "radius_q90": float(np.quantile(radii, 0.90)),
        "radius_pair_scale_median": radius_pair_scale_median,
        "radius_center_eccentricity_correlation": _safe_correlation(
            radii.reshape(-1),
            center_eccentricity.reshape(-1),
        ),
        "sampled_train_estimator_coverage_mean": float(
            np.mean(coverage_by_estimator)
        ),
        "sampled_train_covering_estimators_mean": float(
            np.mean(covering_estimators)
        ),
        "sampled_train_covering_estimator_fraction_mean": float(
            np.mean(covering_estimators / estimator_count)
        ),
        "sampled_train_uncovered_fraction": float(
            np.mean(covering_estimators == 0)
        ),
    }
