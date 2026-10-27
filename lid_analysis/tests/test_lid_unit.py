"""

Unit tests for the MLE-LID implementation.
=> "Does the code woks?"

These tests use small synthetic datasets generated in memory.
They verify code correctness, numerical validity, reproducibility,
and input validation.

They do NOT evaluate LID on the ADERH baseline datasets.

"""

import numpy as np
import pytest

from lid_analysis.lid import estimate_lid

def test_one_lid_per_observation():

    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 5))

    lid = estimate_lid(X, k=20)

    assert lid.shape == (100,)


def test_lid_values_are_valid():

    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 5))

    lid = estimate_lid(X, k=20)

    assert np.all(np.isfinite(lid))
    assert np.all(lid > 0)


def test_lid_is_reproducible():

    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 5))

    lid_1 = estimate_lid(X, k=20)
    lid_2 = estimate_lid(X, k=20)

    assert np.allclose(lid_1, lid_2)


def test_invalid_k():

    rng = np.random.default_rng(42)

    X = rng.normal(size=(10, 5))

    with pytest.raises(ValueError):
        estimate_lid(X, k=20)