import numpy as np
from sklearn.base import BaseEstimator
from util import normalize, ols

class LinearRegression(BaseEstimator):
    """
    Class with simple implementation of linear regression.
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=X.dtype)
        x, x_offset, x_scale = normalize(X)
        y, y_offset = normalize(y, False)

        # Solve for OLS
        coef, residue, rank , sv = ols(x, y)

        self.coef_ = coef / x_scale
        self.intercept_ = y_offset - np.dot(x_offset, self.coef_)

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


