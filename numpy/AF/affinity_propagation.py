import numpy as np


class AffinityPropagation:
    """
    Affinity propagation
    clustering algorithm.
    Also performs k optimization
    by iteration and elbow method.
    """

    def __init__(
        self,
        preference=None,
        damping=0.5,
        max_iter=1000,
        plateau=50,
    ):
        self.preference = preference
        self.damping = damping
        self.max_iter = max_iter
        self.plateau = plateau
        self.similarity_ = None
        self.responsibility_ = None
        self.availability_ = None
        self.labels_ = None
        self.exemplars_ = None

    def _get_similarity_matrix(self, x: np.array) -> np.array:
        def _cosine(a, b):
            return (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

        S = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                S[i, j] = _cosine(x[i], x[j])

        # Set preference value.
        if self.preference is None:
            self.preference = np.median(S)

        np.fill_diagonal(S, self.preference)

        return S

    def fit(self, x: np.array):
        def _update_r():
            # Get summation values for similarity and responsibility.
            values = self.similarity_ + self.availability_
            rows = np.arange(x.shape[0])
            # Fill diagonal with arbitrary min values.
            np.fill_diagonal(values, -np.inf)
            # Get max value for each row in values.
            indices = np.argmax(values, axis=1)
            # First max in each row.
            first = values[rows, indices]
            # Set first max to arbitrary min to get second max.
            values[rows, indices] = -np.inf
            second = values[rows, np.argmax(values, axis=1)]
            # Broadcast to isolate the max values.
            m = np.zeros_like(self.responsibility_) + first[:, None]
            m[rows, indices] = second
            # Update the responsibility matrix.
            update = self.similarity_ - m
            self.responsibility_ = (
                self.responsibility_ * self.damping
                + (1 - self.damping) * update
            )

        def _update_a():
            # Diagonal matrix to exclude identical indices from the sum.
            k = np.arange(x.shape[0])
            a = np.array(self.responsibility_)
            # Clip negative values.
            a[a < 0] = 0
            np.fill_diagonal(a, 0)
            # Compute sum and subtract diagonal.
            a = a.sum(axis=0)
            a = a + self.responsibility_[k, k]
            a = np.ones(self.availability_.shape) * a
            a -= np.clip(self.responsibility_, 0, np.inf)
            # Clip positive values.
            a[a > 0] = 0
            # Compute diagonal availability values.
            w = np.array(self.responsibility_)
            np.fill_diagonal(w, 0)
            # Clip negative values.
            w[w < 0] = 0
            # Compute sum while adding to array.
            a[k, k] = w.sum(axis=0)
            self.availability_ = (
                self.availability_ * self.damping + (1 - self.damping) * a
            )

        def _update_exemplars():
            s = self.availability_ + self.responsibility_
            # Choose each point's max value as the label.
            self.labels_ = np.argmax(s, axis=1)
            self.exemplars_ = np.unique(self.labels_)

        # Build the similarity matrix.
        self.similarity_ = self._get_similarity_matrix(x)
        self.availability_, self.responsibility_ = (
            np.array(self.similarity_),
            np.array(self.similarity_),
        )
        # Iteratively update values until convergence.
        ls = [] + list(range(self.plateau))
        for i in range(self.max_iter):
            _update_r()
            _update_a()
            _update_exemplars()
            if len(set(ls[-self.plateau :])) == 1:
                break

            ls.append(len(self.exemplars_))
