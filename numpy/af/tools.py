import numpy as np
from affinity_propagation import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle


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


def visualize_clusters(clusterer: AffinityPropagation, x: np.ndarray) -> None:
    """
    Visualize clusters and their exemplars by saving a figure
    to the current directory.

    Parameters
    ---------
        clusterer : affinity_propagation.AffinityPropagation
            Clusterer object.
        x : np.array
            Cluster data.
    """
    print(len(clusterer.exemplars_))
    colors = dict(zip(clusterer.exemplars_, cycle("bgrcmyk")))
    for i in range(len(clusterer.labels_)):
        X = x[i][0]
        Y = x[i][1]

        if i in clusterer.exemplars_:
            exemplar = i
            edge = "k"
            ms = 10

        else:
            exemplar = clusterer.labels_[i]
            ms = 3
            edge = None
            plt.plot(
                [X, x[exemplar][0]], [Y, x[exemplar][1]], c=colors[exemplar]
            )

        plt.plot(
            X, Y, "o", markersize=ms, markeredgecolor=edge, c=colors[exemplar]
        )

    plt.savefig("plot.png")
