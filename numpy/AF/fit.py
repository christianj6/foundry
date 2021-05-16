from affinity_propagation import AffinityPropagation
from tools import generate_random_cluster_data
import matplotlib.pyplot as plt
from itertools import cycle

x = generate_random_cluster_data()
af = AffinityPropagation(
    preference=None,
    damping=0.4,
    plateau=50,
)
af.fit(x)

print(len(af.exemplars_))
colors = dict(zip(af.exemplars_, cycle("bgrcmyk")))
for i in range(len(af.labels_)):
    X = x[i][0]
    Y = x[i][1]

    if i in af.exemplars_:
        exemplar = i
        edge = "k"
        ms = 10

    else:
        exemplar = af.labels_[i]
        ms = 3
        edge = None
        plt.plot([X, x[exemplar][0]], [Y, x[exemplar][1]], c=colors[exemplar])

    plt.plot(
        X, Y, "o", markersize=ms, markeredgecolor=edge, c=colors[exemplar]
    )

plt.savefig("plot.png")
