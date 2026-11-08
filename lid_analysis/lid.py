import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_lid(X, k=20, reference=None, eps=1e-12):
    """
    Estimate Local Intrinsic Dimensionality (LID)
    for each observation using the MLE estimator.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Points for which LID will be estimated.

    k : int, default=20
        Number of nearest neighbours used.

    reference : array-like or None
        Reference dataset used to find nearest neighbours.

        If None, X is used as the reference dataset itself.

    eps : float, default=1e-12
        Small value used for numerical stability.

    Returns
    -------
    lid_values : numpy.ndarray of shape (n_samples,)
        Estimated LID value for each observation.
    """

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-dimensional array.")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")

    # -------------------------------------------------
    # Case 1: estimate LID of training points
    # using the training set itself
    # -------------------------------------------------
    if reference is None:
        reference = X
        exclude_self = True
    else:
        reference = np.asarray(reference, dtype=float)
        exclude_self = False

        if reference.ndim != 2:
            raise ValueError("reference must be a 2-dimensional array.")

        if X.shape[1] != reference.shape[1]:
            raise ValueError(
                "X and reference must have the same number of features."
            )

        if not np.all(np.isfinite(reference)):
            raise ValueError(
                "reference contains NaN or infinite values."
            )

    n_reference = reference.shape[0]

    if k <= 1:
        raise ValueError("k must be greater than 1.")

    if exclude_self:
        if k >= n_reference:
            raise ValueError(
                "k must be smaller than the number of observations."
            )

        # Because the first neighbour is the point itself
        n_neighbors = k + 1 

    else:
        if k > n_reference:
            raise ValueError(
                "k cannot be larger than the reference dataset."
            )

        n_neighbors = k

    # -------------------------------------------------
    # Find nearest neighbours
    # -------------------------------------------------
    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="brute"
    )

    knn.fit(reference)

    distances, _ = knn.kneighbors(X)

    # When X is also the reference dataset,
    # the first neighbour is the point itself (distance = 0).
    if exclude_self:
        distances = distances[:, 1:]

    # -------------------------------------------------
    # Numerical stability
    # -------------------------------------------------
    distances = np.maximum(distances, eps)

    # Distance to the k-th neighbour
    r_k = distances[:, -1]

    # -------------------------------------------------
    # MLE-LID
    #
    # LID(x) =
    # -1 / mean(log(r_i / r_k))
    # -------------------------------------------------
    ratios = distances / r_k[:, np.newaxis]

    log_ratios = np.log(ratios)

    denominator = -np.mean(log_ratios, axis=1)

    if np.any(denominator <= eps):
        raise ValueError(
            "Degenerate neighbourhood detected while estimating LID."
        )

    lid_values = 1.0 / denominator

    return lid_values