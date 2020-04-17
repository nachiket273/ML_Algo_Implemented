import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import sys
import os
sys.path.append(os.path.abspath('../DecisionTree'))
from DecisionTree import DecisionTree


class RandomForest(BaseEstimator):
    """
    Simple implementation of Random Forest.
    This class has implementation for Random Forest classifier and regressor.
    Dataset bagging is done by simple numpy random choice with replacement.
    For classification the prediction is by majority vote.
    For regression tree the prediction is averge of all estimator predictions.

    Args:
        n_estimators        Number of base estimators (Decision Trees here)
        max_features        Maximum features to be used to construct tree.
                            Default:
                            -   If classifier, default is square root of total
                                features.
                            -   If regressor, default is total number of features.
        max_depth           The maximum depth to which estimators needs to be constructed.
                            Default: np.inf
        min_samples_split   Minimum number of samples need to present for split at the
                            node.
                            Default: 2
        criterion           criterion to be used for split.
                            For classification tree following criterion are supported:
                                - gini
                                - entropy
                            For regression tree following criterion are supported:
                                - mse (mean squared error)
                                - mae (mean absolute error)
                            Default: gini
        random_seed         random seed value for numpy operations.
                            Default: 0
    """
    def __init__(self, n_estimators, max_features=0, max_depth=np.inf, min_samples_split=2, 
                 criterion='gini', random_seed=0):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_seed = random_seed
        self.idxs = []

        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(DecisionTree(max_depth= self.max_depth, 
                                                    min_samples_split=self.min_samples_split,
                                                    max_features = self.max_features,
                                                    criterion=self.criterion,
                                                    random_seed = self.random_seed))
        self.is_classification_forest = False
        if self.criterion == 'gini' or self.criterion == 'entropy':
            self.is_classification_forest = True
        elif self.criterion == 'mse' or self.criterion == 'mae':
            self.is_classification_forest = False
        else:
            raise Exception("Invalid criterion: {}".format(self.criterion))

    def get_subsets(self, X, y, num=1):
        subsets = []

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)
        num_samples = X.shape[0]
        np.random.shuffle(Xy)
        rng = np.random.default_rng(seed= self.random_seed) 
       
        for _ in range(num):       
            idx = rng.choice(
                range(num_samples),
                size = np.shape(range(int(num_samples)), ),
                replace=True
            )
            subsets.append([X[idx], y[idx]])

        return subsets

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        subsets = self.get_subsets(X, y, self.n_estimators)

        if self.max_features == 0:
            if self.is_classification_forest:
                self.max_features = int(math.sqrt(X.shape[1]))
            else:
                self.max_features = int(X.shape[1])
                        
        # Bagging - choose random features for each estimator
        # if max_features is provided, else use square root of
        # total number of features.
        for i, _ in enumerate(self.trees):
            self.trees[i].max_features = self.max_features
            X_sub, y_sub = subsets[i]
            self.trees[i].fit(X_sub, y_sub)
            
    def predict(self, X):
        all_preds = np.empty((X.shape[0], self.n_estimators))

        for i, tree in enumerate(self.trees):
            preds = tree.predict(X)
            all_preds[:, i] = preds

        y_preds = []
        for preds in all_preds:
            if self.is_classification_forest:
                y_preds.append(np.bincount(preds.astype('int')).argmax())
            else:
                y_preds.append(np.average(preds))
        
        return y_preds