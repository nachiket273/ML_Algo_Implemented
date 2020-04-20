import numpy as np
import pandas as pd
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

        if self.is_classification:
            self.loss = DevianceLoss()
        else:
            self.loss = MSE()

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.is_classification:
            classes = list(np.unique(y))
            self.n_classes = len(classes)
            for j, clss in enumerate(classes):
                trees = []
                init = False
                for i in range(self.n_estimators):
                    y_true = y == clss
                    y_true = y_true.astype(int)
                    if not init:
                        y_pred = np.zeros(X.shape[0], np.float32)
                    gradients = self.loss.grad(y_true, y_pred, k=j)
                    tree = DecisionTree(max_depth=self.max_depth, 
                                max_features=self.max_features,
                                min_samples_split = self.min_samples_split,
                                criterion = self.criterion,
                                random_seed = self.random_seed
                            )
                    tree.fit(X, gradients)
                    preds = tree.predict(X)
                    y_pred -= np.multiply(self.learning_rate, preds)
                    trees.append(tree)
                self.trees.append(trees)
        else:
            y_pred = np.zeros(X.shape[0])
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
                y_pred -= np.multiply(self.learning_rate, preds)
                self.trees.append(tree)

    def predict(self, X_test):
        if self.is_classification:
            y_preds = np.zeros((X_test.shape[0], self.n_classes), np.float32)
            for i, _ in enumerate(self.trees):
                y_pred = np.zeros(X_test.shape[0], np.float32)
                for tr in self.trees[i]:
                    preds = tr.predict(X_test)
                    y_pred -= np.multiply(self.learning_rate, preds)
                y_preds[:, i] = y_pred
            y_preds = np.argmax(y_preds, axis=1)
        else:
            y_preds = np.zeros(X_test.shape[0], np.float32)
            for i,_ in enumerate(self.trees):
                preds = self.trees[i].predict(X_test)
                y_preds -= np.multiply(self.learning_rate, preds)

        return y_preds