import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
import time
from Utils import entropy, gini, mse, mae

class Node():
    """
    A class representing a node in decision tree.

    Args:
        feature_idx Index of the feature column.
                    Default: -1
        threshhold  The threshhold for split for the current node.
                    Default: None
        labels      Labels that belong to current node. Leaf node of tree
                    contains all the labels belong to that leaf node, while 
                    non-leaf nodes have this field None.
        left        Object reference for left child node. For leaf node this
                    field is None.
        right       Object reference for right child node. For leaf node this
                    field is None.
        idxs        Indexes of labels present at the node.
                    Default: []
        value       Value of the node.
                    This is used with gradient boost classifier where value is 
                    calculated based on loss function.
                    For other cases , mostly this will be None.
                    Default: None
    """   
    def __init__(self, feature_idx=-1, threshold=None, labels=None, left=None, 
                    right=None, idxs = [], value = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right
        self.idxs = idxs
        self.value = value

class DecisionTree(BaseEstimator):
    """
    Class that represents simple implementation of Decision Tree.
    This class has implementation for both classification tree and regression tree.
    For classification the prediction is majority probability class.
    For regression tree the prediction is averge of all the node labels.

    Args:
        max_depth           The maximum depth to which tree needs to be constructed.
                            Default: np.inf
        min_samples_split   Minimum number of samples need to present for split at the
                            node.
                            Default: 2
        max_features        Maximum features to be used to construct tree.
                            Default: 0
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
    def __init__(self, max_depth=np.inf, min_samples_split=2, max_features=0,
                 criterion='gini', random_seed =0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.root = None
        self.n_features = 0
        self.n_samples = 0
        self.min_impurity = 1e-7
        self.random_seed = random_seed
        self.is_classification_tree = False

        if self.criterion == 'entropy' or self.criterion == 'gini':
            self.is_classification_tree = True

    def randomize(self, feat_arr): 
        n = len(feat_arr)
        for i in range(n-1,0,-1): 
            j = np.random.randint(0,i+1) 
            feat_arr[i], feat_arr[j] = feat_arr[j], feat_arr[i]
        if self.max_features == 0:
            return feat_arr
        return feat_arr[:self.max_features]
        
    def _calc_critn(self, y, yleft, yright):      
        if self.criterion == 'entropy':
            critn = entropy(y)
            critn_left = entropy(yleft)
            critn_right = entropy(yright)
        elif self.criterion == 'gini':
            critn = gini(y)
            critn_left = gini(yleft)
            critn_right = gini(yright)
        elif self.criterion == 'mse':
            critn = mse(y)
            critn_left = mse(yleft)
            critn_right = mse(yright)
        elif self.criterion == 'mae':
            critn = mae(y)
            critn_left = mae(yleft)
            critn_right = mae(yright)
        else:
            raise Exception("Invalid criterion.")
            
        fn = (len(y) * critn - len(yleft) * critn_left - len(yright) * critn_right) / self.n_samples
        return fn
        
    def _get_on_crit(self, Xy, thresh, idx, inds):
        if isinstance(thresh, int) or isinstance(thresh, float):
            right = np.where(Xy[:, idx] >= thresh)
            left = np.where(Xy[:, idx] < thresh)
        else:
            right = np.where(Xy[:, idx] == thresh)
            left = np.where(Xy[:, idx] != thresh)
        
        xy_thresh = Xy[right]
        xynot_thresh = Xy[left]

        return xy_thresh, xynot_thresh, inds[right], inds[left]
        
    def _build_tree(self, X, y, current_depth, inds):
        max_critn = -np.inf
        leftx, lefty , rightx, righty = [],[],[],[]
        left_inds, right_inds = [], []
        threshold = -1
        idx = -1

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)


        feats = self.randomize(list(range(n_features)))

        if current_depth < self.max_depth and n_samples > self.min_samples_split:
            for i in feats:
                feature_values = np.expand_dims(X[:, i], axis=1)
                for thresh in np.unique(feature_values):
                    xy_thresh, xynot_thresh, xy_thresh_inds, xynot_thresh_inds = self._get_on_crit(Xy, thresh, i, inds)
                    if len(xy_thresh) > 0 and len(xynot_thresh) > 0 :
                        x_thresh = xy_thresh[:, :n_features]
                        y_thresh = xy_thresh[:, n_features:]
                        xnot_thresh = xynot_thresh[:, :n_features]
                        ynot_thresh = xynot_thresh[:, n_features:]

                        # find out value of function to be maximized
                        fn = self._calc_critn(y, ynot_thresh, y_thresh)

                        if fn > max_critn:
                            max_critn = fn
                            leftx, lefty, rightx, righty = xnot_thresh, ynot_thresh, x_thresh, y_thresh
                            left_inds, right_inds = xynot_thresh_inds, xy_thresh_inds
                            threshold = thresh
                            idx = i
                                            
            if max_critn > self.min_impurity:
                node = Node(idx, threshold)
                node.left = self._build_tree(leftx, lefty, current_depth+1, left_inds)
                node.right = self._build_tree(rightx, righty, current_depth+1, right_inds)
                return node
        
        return Node(labels=y, idxs=inds)
    
    def fit(self, X, y):
        np.random.seed(self.random_seed)
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.n_features = X.shape[1]
        self.n_samples = X.shape[0]
        inds = np.array(list(range(self.n_samples)))
        self.root = self._build_tree(X, y ,0, inds)
        
    def get_pred(self, node):
        if self.is_classification_tree:
            unique, counts = np.unique(node.labels, return_counts=True)
            return unique[np.argmax(counts)]
        else:
            return np.average(node.labels)

    def calc_value_(self, y, grads, num_class, node=None):
        if not node:
            node = self.root

        if node.labels is not None:
            inds = node.idxs
            num = (num_class - 1) * np.sum(grads[inds]) / num_class
            p = y[inds] - grads[inds]
            denom = np.sum(p * (1-p))
            eps = np.finfo(np.float64).eps
            if denom < eps:
                node.value = 0.0
            else:
                node.value = num/denom
        else:
            self.calc_value_(y, grads, num_class, node.left)
            self.calc_value_(y, grads, num_class, node.right)
                   
    def get_val(self, p, node=None):
        if node is None:
            node = self.root
            
        if node.labels is not None:
            if node.value:
                return node.value
            else:
                return self.get_pred(node)
        
        next_node = node.left
        thresh = node.threshold

        if p[node.feature_idx] >= thresh:
            next_node = node.right
            
        return self.get_val(p, next_node)
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        preds = [self.get_val(p) for p in X]
        return preds
    
    def get_probs(self, p, node=None):
        if node is None:
            node = self.root
            
        probs = dict()
            
        if node.labels is not None:
            unique, counts = np.unique(node.labels, return_counts=True)
            tot = sum(counts)
            for i, j in zip(unique, counts):
                probs[i] = j/tot
            return probs
        
        next_node = node.left
        
        if p[node.feature_idx] >= node.threshold:
            next_node = node.right
            
        return self.get_probs(p, next_node)
        
    def predict_proba(self, X):
        # only valid for classification
        if not self.is_classification_tree:
            raise Exception("Invalid Operation for criterion {}".format(self.criterion))

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        probs = [self.get_val(p) for p in X]
        return probs

    def display(self, node=None):
        if not node:
            node = self.root

        if node.labels is not None:
            print("Num Labels:: {}".format(len(node.labels)))
            print ("Labels:: {}".format(node.labels))
        else:
            print ("Index: {}".format(node.feature_idx))
            print ("Threshold: {}".format(node.threshold))
            self.display(node.left)
            self.display(node.right)