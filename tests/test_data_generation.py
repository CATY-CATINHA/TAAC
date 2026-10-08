import numpy as np

from aderh_distance.data_generation import DEFAULT_DIMENSIONS, make_nested_datasets


def test_default_shapes_and_exact_anomaly_proportion() -> None:
    datasets = make_nested_datasets(random_state=7)

    assert tuple(datasets) == DEFAULT_DIMENSIONS
    for dimensionality, dataset in datasets.items():
        assert dataset.X.shape == (2_000, dimensionality)
        assert dataset.y.shape == (2_000,)
        assert int(dataset.y.sum()) == 100
        assert dataset.y.mean() == 0.05


def test_features_are_fixed_prefixes_across_dimensionalities() -> None:
    datasets = make_nested_datasets((2, 5, 50, 500), random_state=19)

    assert np.array_equal(datasets[2].X, datasets[500].X[:, :2])
    assert np.array_equal(datasets[5].X, datasets[500].X[:, :5])
    assert np.array_equal(datasets[50].X, datasets[500].X[:, :50])
    assert np.array_equal(datasets[2].y, datasets[500].y)


def test_prefixes_do_not_depend_on_requested_maximum_dimension() -> None:
    smaller = make_nested_datasets((2, 50), random_state=23)
    larger = make_nested_datasets((2, 50, 500), random_state=23)

    assert np.array_equal(smaller[2].X, larger[2].X)
    assert np.array_equal(smaller[50].X, larger[50].X)


def test_generation_is_reproducible_and_seed_sensitive() -> None:
    first = make_nested_datasets((10,), random_state=5)[10]
    repeated = make_nested_datasets((10,), random_state=5)[10]
    different = make_nested_datasets((10,), random_state=6)[10]

    assert np.array_equal(first.X, repeated.X)
    assert np.array_equal(first.y, repeated.y)
    assert not np.array_equal(first.X, different.X)

