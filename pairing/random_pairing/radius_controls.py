"""Random-pairing controls that isolate center/radius interactions."""

from __future__ import annotations

import numpy as np
from aderh import ADERH
from sklearn.utils import check_random_state

from pairing.common.coverage import (
    coverage_at_scale,
    coverage_scale_thresholds,
    sample_coverage_indices,
    scale_for_target_coverage,
)

from .corrected_random_aderh import CorrectedRandomADERH


class ShuffledRadiusRandomADERH(CorrectedRandomADERH):
    """Preserve radius values but randomly reassign them between pairs.

    Centers and partner selection are identical to ``CorrectedRandomADERH``.
    Within each estimator, the squared pair radii are permuted before being
    duplicated in blocked center order. This deliberately breaks the direct
    correlation between a pair's location/distance and its assigned radius
    while preserving the exact per-estimator radius distribution.
    """

    def _transform_pair_radii(
        self,
        pair_radii,
        *,
        X,
        anchor_idx,
        partner_idx,
        rnd,
    ):
        del X, anchor_idx, partner_idx
        return np.asarray(pair_radii)[rnd.permutation(len(pair_radii))]


def _calibrate_to_original_training_coverage(model, X) -> None:
    """Scale ``model`` radii to sampled coverage of official ADERH."""

    if model.calibration_max_points < 1:
        raise ValueError("calibration_max_points must be positive")
    if model.calibration_max_estimators < 1:
        raise ValueError("calibration_max_estimators must be positive")

    reference = ADERH(
        n_estimators=model.n_estimators,
        n=model.n,
        contamination=model.contamination,
        random_state=model.random_state,
    ).fit(X)

    sampling_random = check_random_state(model.random_state)
    sampling_seed = int(
        sampling_random.randint(np.iinfo(np.int32).max)
    )
    point_indices, estimator_indices = sample_coverage_indices(
        n_points=X.shape[0],
        n_estimators=model.n_estimators,
        random_state=sampling_seed,
        max_points=model.calibration_max_points,
        max_estimators=model.calibration_max_estimators,
    )

    reference_thresholds = coverage_scale_thresholds(
        X,
        reference._centers,
        reference._sq_radii,
        point_indices=point_indices,
        estimator_indices=estimator_indices,
    )
    candidate_thresholds = coverage_scale_thresholds(
        X,
        model._centers,
        model._sq_radii,
        point_indices=point_indices,
        estimator_indices=estimator_indices,
    )

    target_coverage = coverage_at_scale(reference_thresholds, 1.0)
    base_coverage = coverage_at_scale(candidate_thresholds, 1.0)
    requested_gamma = scale_for_target_coverage(
        candidate_thresholds,
        target_coverage,
    )
    # Calibration is a coverage-recovery policy, not a radius-shrinkage
    # policy. Preserve the corrected base geometry when it already meets the
    # reference target.
    gamma = max(1.0, requested_gamma)

    model._sq_radii *= gamma
    model.radius_scale_gamma_ = float(gamma)
    model.calibration_target_coverage_ = float(target_coverage)
    model.calibration_base_coverage_ = float(base_coverage)
    model.calibration_achieved_coverage_ = coverage_at_scale(
        candidate_thresholds,
        gamma,
    )
    model.calibration_scale_floor_applied_ = bool(requested_gamma < 1.0)
    model.calibration_point_count_ = int(point_indices.size)
    model.calibration_estimator_count_ = int(estimator_indices.size)


class CoverageCalibratedRandomADERH(CorrectedRandomADERH):
    """Corrected random pairing calibrated to original training coverage."""

    def __init__(
        self,
        n_estimators=256,
        n=18,
        contamination=0.1,
        random_state=None,
        calibration_max_points=256,
        calibration_max_estimators=64,
    ):
        super().__init__(
            n_estimators=n_estimators,
            n=n,
            contamination=contamination,
            random_state=random_state,
        )
        self.calibration_max_points = calibration_max_points
        self.calibration_max_estimators = calibration_max_estimators

    def _postprocess_geometry(self, X):
        _calibrate_to_original_training_coverage(self, X)


class ShuffledCoverageCalibratedRandomADERH(ShuffledRadiusRandomADERH):
    """Shuffled radii followed by original-coverage calibration."""

    def __init__(
        self,
        n_estimators=256,
        n=18,
        contamination=0.1,
        random_state=None,
        calibration_max_points=256,
        calibration_max_estimators=64,
    ):
        super().__init__(
            n_estimators=n_estimators,
            n=n,
            contamination=contamination,
            random_state=random_state,
        )
        self.calibration_max_points = calibration_max_points
        self.calibration_max_estimators = calibration_max_estimators

    def _postprocess_geometry(self, X):
        _calibrate_to_original_training_coverage(self, X)
