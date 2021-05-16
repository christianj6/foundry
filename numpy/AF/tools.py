import numpy as np


def generate_random_cluster_data(n=5, m=10) -> np.array:
    """
    Generates a random dataset of
    dummy clustering data based on
    the number of clusters provided.

    Parameters
    ---------
        n : int
            Number of desired clusters.
        m : int
            Number of points per cluster.

    Returns
    ---------
        X : np.array
            Data.
    """
    size = (m, 2)
    X = []
    for i in range(n):
        center = np.random.rand(2) * 35
        x = np.random.normal(0, 1, size)
        x = np.append(x, np.random.normal(center, 0.5, size), axis=0)
        X.append(x)

    return np.array(X).reshape((-1, 2))
