"""
fit.py — Affinity Propagation Demo
====================================

Generates a random 2D dataset with several natural clusters, runs the
from-scratch AffinityPropagation algorithm, and saves a visualization of
the resulting cluster assignments to `plot.png`.

The preference parameter is left as None (defaulting to the median
similarity), which lets the algorithm self-select an appropriate number
of clusters from the data.
"""

from affinity_propagation import AffinityPropagation
from tools import generate_random_cluster_data, visualize_clusters

x = generate_random_cluster_data()
af = AffinityPropagation(
    preference=None,
    damping=0.9,
    plateau=50,
)
af.fit(x)
visualize_clusters(af, x)
