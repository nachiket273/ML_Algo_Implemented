import numpy as np
from scipy.special import logsumexp

class Loss(object):
    def loss(self, y_true, y_pred):
        raise NotImplementedError()

    def grad(self, y_true, y_pred):
        raise NotImplementedError()

class DevianceLoss(Loss):
    def loss(self, y_true, y_pred):
        n_classes = len(np.unique(y_true))

        Y = np.zeros((y_true.shape[0], n_classes), dtype=np.float64)
        for i in range(n_classes):
            Y[:, i] = y_true == i

        return np.average(-1 * (y_true * y_pred).sum(axis=1) + logsumexp(y_pred, axis=1))

    def grad(self, y_true, y_pred, k=0):
        return y_true - np.nan_to_num(np.exp(y_pred[:, k] -
                                        logsumexp(y_pred, axis=1)))

class MSE(Loss):
    def loss(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred.ravel()) ** 2.0)

    def grad(self, y_true, y_pred):
        return y_true - y_pred.ravel()