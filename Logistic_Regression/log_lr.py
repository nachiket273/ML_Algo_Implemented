import math
import numpy as np
from sklearn.base import BaseEstimator
from util import Softmax, CrossEntropy, Sigmoid, BinaryCrossEntropy

class LogisticRegression(BaseEstimator):
    """
    Class with simple implementation of logistic regression.
    This implementation uses Gradient Descent for optimization.
    The implementation uses sigmoid and binary cross-entropy for binary
    classification, while softmax and general cross entropy loss for 
    multiclass classification.

    Args:
        lr                  Learning rate to scale gradient to update weights and biases.
                            Default: 0.1
        regularization      Regularization methods(l1 - lasso or l2 - ridge)
                            Default: l2
        C                   The inverse for regualrization strength.
                            Note that smaller values specify stronger regularization.
                            Default: 1
        iters               Number of epoachs for which gradient descent is run.
                            Default: 100
        random_seed         random seed value for numpy operations.
                            Default: 0
    """
    def __init__(self, lr=0.1, regularization='l2', C= 1.0, iters=100, random_seed=0):
        self.lr = lr
        self.regularization = regularization
        assert(self.regularization == 'l1' or self.regularization == 'l2')
        self.C = C
        self.iters = iters
        self.random_seed = random_seed
        self.weights_ = None
        self.biases_ = None
        self.classes_ = 0
        self.act_ = Softmax()
        self.cross_entropy_ = CrossEntropy()
        self.losses = []

    def init_weights(self, num_features):
        gen = np.random.RandomState(self.random_seed)
        return gen.normal(loc=0.0, scale=0.01, size=(num_features, self.classes_))

    def init_biases(self):
        return np.zeros((self.classes_,))

    def one_hot(self, y):
        one_hot_vec = np.zeros((len(y), self.classes_), dtype=np.float)
        for i, val in enumerate(y):
            one_hot_vec[i, val] = 1
        return one_hot_vec

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        self.classes_ = len(np.unique(y))
        if self.classes_ == 2:
            self.classes_ = 1
            self.act_ = Sigmoid()
            self.cross_entropy_ = BinaryCrossEntropy()
            y = y[:, np.newaxis]
        self.weights_ = self.init_weights(X.shape[1])
        self.biases_  = self.init_biases()

        # One hot encode the labels
        if self.classes_ > 2:
            y_onehot = self.one_hot(y)

            for i in range(self.iters):
                ypreds = self.act_.loss(np.dot(X, self.weights_) + self.biases_)
                grads = self.cross_entropy_.grad(y_onehot, ypreds)
                self.biases_ -= self.lr * np.sum(grads, axis=0)
                if self.regularization == 'l2':
                    grads = np.dot(np.transpose(X), grads) + 1.0 * self.weights_ / self.C
                else :
                    grads = np.dot(np.transpose(X), grads) + 1.0 / self.C

                self.weights_ -= self.lr * grads

                loss = self.cross_entropy_.loss(y_onehot, ypreds)
                loss = np.mean(loss)
                loss += 0.5 * np.sum(self.weights_**2) / self.C
                self.losses.append(loss)
                print("Epoch: {}, Loss: {}".format(i+1, loss))

        else:
            for i in range(self.iters):
                ypreds = self.act_.loss(np.dot(X, self.weights_) + self.biases_)
                grads = self.cross_entropy_.grad(y, ypreds)
                self.biases_ -= self.lr * np.sum(grads, axis=0)
                if self.regularization == 'l2':
                    grads = np.dot(np.transpose(X), grads) + 1.0 * self.weights_ / self.C
                else :
                    grads = np.dot(np.transpose(X), grads) + 1.0 / self.C

                self.weights_ -= self.lr * grads

                loss = self.cross_entropy_.loss(y, ypreds)
                loss = np.mean(loss)
                loss += 0.5 * np.sum(self.weights_**2) / self.C
                self.losses.append(loss)
                print("Epoch: {}, Loss: {}".format(i+1, loss))

    def predict_proba(self, X):
        return self.act_.loss(np.dot(X, self.weights_) + self.biases_)

    def predict(self, X):
        if self.classes_ > 2:
            return np.argmax(self.predict_proba(X), axis=1)
        else:
            return np.where(self.predict_proba(X) < 0.5, 0, 1)
        