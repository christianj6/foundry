from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Trainer:
    """
    Object for managing model training
    and tuning.
    """

    def __init__(
        self,
        keyword: str,
        data: list,
        labels: list,
    ):
        """
        Instantiate the trainer,
        fitting a new model.

        Parameters
        ---------
            language : str
                Source language.
            keyword : str
                Keyword for which
                we train the model.
            data : list
                List of arrays which
                are the feature-arrays
                used to train the model.
            labels : list
                List of same size as the data
                containing training labels.
        """
        self.keyword = keyword
        self.data = data
        self.labels = labels
        # Fit the model to the initial data.
        self.train()

    def train(self):
        """
        Fit a model to the input
        training data based on the
        input labels.

        Returns
        ---------
            model : sklearn.svm.LinearSVC
                Fitted classifier.
        """
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                random_state=42,
            ),
        )
        clf.fit(self.data, self.labels)
        # Return trained model.
        self.model = clf

    def predict(self, X):
        """
        Get a prediction from
        the fitted model.

        Parameters
        ---------
            X : np.array
                Test point for inference.

        Returns
        ---------
            pred : int
                Binary prediction.
        """
        return self.model.predict(X)[0]
