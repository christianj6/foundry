# Numpy Machine Learning

Machine learning algorithms implemented from scratch using only Python and NumPy — no high-level frameworks.

Each algorithm is written as a standalone class that closely mirrors the scikit-learn interface (`fit`, `predict`/`eval`), making the implementations easy to compare against their library equivalents.

***

## Content

### `nn/`

A configurable feed-forward neural network with full backpropagation implemented by hand. Trained and evaluated on the MNIST handwritten digit dataset. Demonstrates gradient descent, mini-batching, sigmoid activations, and one-hot encoding — all without a framework.

### `af/`

The Affinity Propagation clustering algorithm, built from scratch. Iteratively updates responsibility and availability matrices until cluster exemplars converge. Includes random cluster data generation and a matplotlib visualization showing each point connected to its exemplar.

***
