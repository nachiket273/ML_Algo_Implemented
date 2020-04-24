import numpy as np
import pandas as pd
from scipy.special import logsumexp 
from sklearn.base import BaseEstimator
from Loss import DevianceLoss, MSE

import sys
import os
sys.path.append(os.path.abspath('../DecisionTree'))
from DecisionTree import DecisionTree

class GradientBoost(BaseEstimator):
    def __init__(self, n_estimators, learning_rate=0.1, criterion='mse', 
                    max_features=0, 
                    max_depth=3, min_samples_split=2, 
                    is_classification=True, random_seed=0):
        self.n_estimators = n_estimators
        self.trees = []
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.max_depth = max_depth
        self.criterion = criterion
        assert(criterion == 'mse' or criterion == 'mae')
        self.min_samples_split = min_samples_split
        self.random_seed = random_seed
        self.is_classification = is_classification
        self.constant_ = 0
        self.priors_ = None
        self.n_classes = 2

        if self.is_classification:
            self.loss = DevianceLoss()
        else:
            self.loss = MSE()

    def init_priors_(self, y):
        if self.priors_ is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
            counts = np.bincount(y, weights=sample_weight)
            priors = counts/sum(counts)
            self.priors_ = priors
        p = []
        for i in range(self.n_classes):
            out = np.ones((y.shape[0], 1)) * self.priors_[i]
            p.append(out)
        raw_preds = np.log(p).astype(np.float64)
        return np.transpose(np.squeeze(raw_preds))

    def preds_to_probs(self, preds):
        return np.nan_to_num(
            np.exp(preds - (logsumexp(preds, axis=1)[:, np.newaxis])))

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.is_classification:
            classes = list(np.unique(y))
            self.n_classes = len(classes)
            y_pred = self.init_priors_(y)

            for i in range(self.n_estimators):
                trees = []
                for j, clss in enumerate(classes):
                    y_true = np.array(y == clss, dtype=np.float64)
                    gradients = self.loss.grad(y_true, y_pred, k=j)
                    tree = DecisionTree(max_depth=self.max_depth, 
                                max_features=self.max_features,
                                min_samples_split = self.min_samples_split,
                                criterion = self.criterion,
                                random_seed = self.random_seed
                            )
                    tree.fit(X, gradients)
                    tree.calc_value_(y_true, gradients, self.n_classes)
                    preds = tree.predict(X)
                    y_pred[:, j] += np.multiply(self.learning_rate, preds)
                    trees.append(tree)
                self.trees.append(trees)
        else:
            self.constant_ = np.mean(y)
            y_pred = np.full(X.shape[0], self.constant_)
            for i in range(self.n_estimators):
                gradients = self.loss.grad(y, y_pred)
                tree = DecisionTree(max_depth=self.max_depth, 
                                max_features=self.max_features,
                                min_samples_split = self.min_samples_split,
                                criterion = self.criterion,
                                random_seed = self.random_seed
                            )
                tree.fit(X, gradients)
                preds = tree.predict(X)
                y_pred += np.multiply(self.learning_rate, preds)
                self.trees.append(tree)

    def predict(self, X_test):
        if self.is_classification:
            y_preds = self.init_priors_(X_test)
            for i, _ in enumerate(self.trees):
                for j, tr in enumerate(self.trees[i]):
                    preds = tr.predict(X_test)
                    y_preds[:, j] += np.multiply(self.learning_rate, preds)

            y_preds = self.preds_to_probs(y_preds)
            y_preds = np.argmax(y_preds, axis=1)
        else:
            y_preds = np.full(X_test.shape[0], self.constant_)
            for i,_ in enumerate(self.trees):
                preds = self.trees[i].predict(X_test)
                y_preds += np.multiply(self.learning_rate, preds)

        return y_preds