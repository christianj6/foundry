from affinity_propagation import AffinityPropagation
from tools import generate_random_cluster_data, visualize_clusters

x = generate_random_cluster_data()
af = AffinityPropagation(
    preference=None,
    damping=0.4,
    plateau=50,
)
af.fit(x)
visualize_clusters(af, x)
