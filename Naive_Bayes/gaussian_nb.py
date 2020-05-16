import numpy as np

class GaussianNB():
    '''
    A class with simple implementation of Gaussian Naive Bayes.

    Args:
        smoothing   Smoothning factor for variance to avoid divide by zero.
                    Default: 1e-9
    '''
    def __init__(self, smoothing= 1e-9):
        self.smoothing = smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        num_samples, num_features = X.shape

        self.mean_ = np.zeros((num_classes, num_features))
        self.var_  = np.zeros((num_classes, num_features))
        self.counts_ = np.zeros(num_classes)

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[i, :] = np.mean(X_c, axis=0)
            self.var_[i, :] = np.var(X_c, axis=0)
            self.counts_[i] = X_c.shape[0]

        self.var_[:,:] += self.smoothing
        self.priors_ = self.counts_ / self.counts_.sum()

    def _calc_log_likelihood(self, X_test):
        likelihood = []

        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            gaus = -0.5 * np.sum(np.log(2 * np.pi * self.var_[i, :]))
            gaus -= 0.5 * np.sum(((X_test - self.mean_[i, :] )** 2) / (self.var_[i, :]), axis= 1)
            likelihood.append(log_prior+gaus)

        return np.array(likelihood).T


    def predict(self, X_test):
        y_preds = self._calc_log_likelihood(X_test)
        return self.classes_[np.argmax(y_preds, axis=1)]