"""

Synthetic validation tests for the MLE-LID estimator.
=> "Does the estimator behave as expected on synthetic data?"

These tests use Gaussian datasets with known dimensionality.
Goal:
- verify that estimated LID increases when the true dimensionality increases;
- verify that the estimator gives reasonable values in low/moderate dimensions.

These are NOT unit tests of code mechanics and do NOT use ADERH baseline data.

"""

import numpy as np

from lid_analysis.lid import estimate_lid


def generate_gaussian_data(n_samples, n_dimensions, seed=42):
    """
    Generate synthetic Gaussian data with a known ambient/intrinsic dimension.
    """
    rng = np.random.default_rng(seed)

    return rng.normal(
        size=(n_samples, n_dimensions)
    )


def test_lid_increases_with_dimension():
    """
    LID should increase when the dimensionality
    of the synthetic data increases.
    """

    dimensions = [2, 5, 10, 20]

    median_lids = []

    for d in dimensions:

        X = generate_gaussian_data(
            n_samples=2000,
            n_dimensions=d,
            seed=42 + d
        )

        lid = estimate_lid(X, k=20)

        median_lid = np.median(lid)

        median_lids.append(median_lid)

    # Check monotonic increase
    assert all(
        median_lids[i] < median_lids[i + 1]
        for i in range(len(median_lids) - 1)
    )


def test_lid_is_reasonable_for_low_dimensions():
    """
    For low-dimensional Gaussian data,
    estimated LID should be reasonably close
    to the known dimensionality.
    """

    X_2d = generate_gaussian_data(
        n_samples=2000,
        n_dimensions=2,
        seed=42
    )

    lid_2d = estimate_lid(X_2d, k=20)

    median_lid_2d = np.median(lid_2d)

    assert 1.0 < median_lid_2d < 4.0


def test_lid_values_remain_valid_across_dimensions():
    """
    Check that LID remains finite and positive
    across several dimensionalities.
    """

    dimensions = [2, 5, 10, 20]

    for d in dimensions:

        X = generate_gaussian_data(
            n_samples=1000,
            n_dimensions=d,
            seed=42 + d
        )

        lid = estimate_lid(X, k=20)

        assert np.all(np.isfinite(lid))
        assert np.all(lid > 0)