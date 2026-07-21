### Foundry

An archive of early Python programming and machine learning experiences — ranging from async I/O benchmarks and from-scratch ML algorithms to a published fuzzy-search package and a minimal language model.

***

#### Packages

| Package | Description |
|---|---|
| [fast-keyword-identification](packages/fast-keyword-identification/) | Fuzzy keyword search over large document collections using character n-gram TF-IDF vectors and approximate cosine similarity. Ships with a CLI, an optional LinearSVC classification filter, and a demo corpus of airline tweets. |

***

#### Experiments

| Experiment | Description |
|---|---|
| [asyncio](experiments/asyncio/) | Side-by-side benchmarks of synchronous vs. asynchronous Python for HTTP requests and file writes. Demonstrates where `asyncio` wins and where it does not. |
| [numpy/nn](experiments/numpy/nn/) | Feed-forward neural network built from scratch with NumPy. Backpropagation implemented by hand; trained and evaluated on MNIST. |
| [numpy/af](experiments/numpy/af/) | Affinity propagation clustering algorithm built from scratch with NumPy. Includes random cluster data generation and matplotlib visualization. |
