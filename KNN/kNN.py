import numpy as np
from sklearn.base import BaseEstimator

def get_eucliedean_dist(x, y):
    return np.sqrt(np.sum(y**2, axis=1).reshape(y.shape[0], 1) \
    + np.sum(x**2, axis=1) - 2 * y.dot(x.T))


class kNN(BaseEstimator):
    """
    Implementation of k-nearest neighbor algorithm.

    Args:
        k           Number of neighbors to be considered for prediction.
                    Default: 5

        classifier  Boolean variable to decide if it's a classifier or regressor.
                    If false, the predict function will return mean instead of 
                    majority class.
                    Default: True
    """
    def __init__(self, k=5, classifier=True):
        self.k = k
        self.classifier = classifier

    def fit(self, X, y):
        assert(self.k < X.shape[0])
        self.X = X
        self.y = y

    def predict(self, x):
        dists = get_eucliedean_dist(self.X, x)

        # get index of first k sorted predictions
        idxs = np.argsort(dists)[:, :self.k]

        preds = list()

        # use these idx to find labels
        for i, idx in enumerate(idxs):
            labels = self.y[idx]
            if self.classifier:
                cnts = np.bincount(labels.reshape(labels.shape[0]))
                preds.append(np.argmax(cnts))
            else:
                mean = np.mean(labels.reshape(labels.shape[0]))
                preds.append(mean)

        return preds