import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from Loss import CrossEntropyLoss, MSE

import sys
import os
sys.path.append(os.path.abspath('../DecisionTree'))
from DecisionTree import DecisionTree

class GradientBoost(BaseEstimator):
    def __init__(self, n_estimators, learning_rate=0.1, max_features=0, 
                    max_depth=np.inf, min_samples_split=2, 
                    is_classification=True, random_seed=0):
        self.n_estimators = n_estimators
        self.trees = []
        self.learning_rate = learning_rate
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_seed = random_seed
        self.is_classification = is_classification

        if self.is_classification:
            self.loss = CrossEntropyLoss()
        else:
            self.loss = MSE()

        for _ in range(self.n_estimators):
            self.trees.append(DecisionTree(max_depth=self.max_depth, 
                                max_features=self.max_features,
                                min_samples_split = self.min_samples_split,
                                criterion = 'mse',
                                random_seed = self.random_seed
                            ))

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        y_pred = np.full(X.shape[0], np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradients = self.loss.grad(y, y_pred)

            # Now fit the regression tree with this gradient as 
            # target variables.
            self.trees[i].fit(X, gradients)

            # Make predictions on same training data
            preds = self.trees[i].predict(X)

            # Now use predictions and learning rate to mimimize
            # error.
            y_pred -= np.multiply(self.learning_rate, preds)

    def predict(self, X_test):
        y_preds = np.array([])
        for i,_ in enumerate(self.trees):
            preds = self.trees[i].predict(X_test)
            if y_preds.any():
                y_preds -= np.multiply(self.learning_rate, preds)
            else:
                y_preds = -np.multiply(self.learning_rate, preds)

        if self.is_classification:
            y_preds = np.expand_dims(y_preds, axis=1)
            exp_preds = np.exp(y_preds)
            exp_preds = exp_preds/ np.expand_dims(np.sum(exp_preds, axis=1), axis=1)
            y_preds = np.argmax(exp_preds, axis=1)

        return y_preds