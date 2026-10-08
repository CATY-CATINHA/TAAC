import numpy as np
import pytest

from aderh_distance.metrics import relative_contrast


def test_relative_contrast_is_one_for_equidistant_reference_points() -> None:
    equilateral_triangle = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) / 2.0],
        ]
    )

    result = relative_contrast(
        equilateral_triangle,
        random_state=11,
        reference_sample_size=3,
        random_pair_count=100,
    )

    assert result.value == pytest.approx(1.0)
    assert result.mean_random_distance == pytest.approx(1.0)
    assert result.mean_nearest_neighbor_distance == pytest.approx(1.0)
    assert result.reference_sample_size == 3
    assert result.random_pair_count == 100


def test_relative_contrast_is_scale_invariant() -> None:
    rng = np.random.default_rng(31)
    X = rng.normal(size=(100, 7))

    original = relative_contrast(X, random_state=2)
    rescaled = relative_contrast(9.5 * X, random_state=2)

    assert rescaled.value == pytest.approx(original.value)

