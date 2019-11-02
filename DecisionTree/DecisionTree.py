import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from Utils import entropy, gini, variance, mad_median

class Node():   
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right

class DecisionTree(BaseEstimator):
    
    def __init__(self, max_depth=np.inf, min_samples_split=2, 
                 criterion='gini', debug=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.debug = debug
        self.root = None
        self.min_impurity = 1e-7
        
    def _calc_critn(self, y, yleft, yright):      
        if self.criterion == 'entropy':
            critn = entropy(y)
            critn_left = entropy(yleft)
            critn_right = entropy(yright)
        elif self.criterion == 'gini':
            critn = gini(y)
            critn_left = gini(yleft)
            critn_right = gini(yright)
        elif self.criterion == 'variance':
            critn = variance(y)
            critn_left = variance(yleft)
            critn_right = variance(yright)
        else:
            critn = mad_median(y)
            critn_left = mad_median(yleft)
            critn_right = mad_median(yright)
            
        fn = critn - len(yleft) * critn_left / len(y) - len(yright) * critn_right / len(y)        
        return fn
        
    def _get_on_crit(self, Xy, thresh, idx):
        if isinstance(thresh, int) or isinstance(thresh, float):
            xy_thresh = [p for p in Xy if p[idx] >= thresh]
            xynot_thresh = [p for p in Xy if p[idx] < thresh]
        else:
            xy_thresh = [p for p in Xy if p[idx] == thresh]
            xynot_thresh = [p for p in Xy if not p[idx] == thresh]
        return np.array(xy_thresh), np.array(xynot_thresh)
        
    def _build_tree(self, X, y, current_depth):
        max_critn = 0
        leftx, lefty , rightx, righty = [],[],[],[]
        threshold = -1
        idx = -1
        
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        
        if current_depth < self.max_depth and n_samples > self.min_samples_split:
            for i in range(n_features):
                feature_values = np.expand_dims(X[:, i], axis=1)
                for thresh in np.unique(feature_values):
                    xy_thresh, xynot_thresh = self._get_on_crit(Xy, thresh, i)
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
                            threshold = thresh
                            idx = i
                            
        if max_critn > self.min_impurity:
            node = Node(idx, threshold)
            node.left = self._build_tree(leftx, lefty, current_depth+1)
            node.right = self._build_tree(rightx, righty, current_depth+1)
            return node
        
        return Node(labels=y)
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y ,0)
        
    def get_pred(self, node):
        if self.criterion == 'gini' or self.criterion == 'entropy':
            unique, counts = np.unique(node.labels, return_counts=True)
            count = 0
            majority = -1
            for i, j in zip(unique, counts):
                if j > count:
                    count = j
                    majority = i
            return majority
        else:
            return np.average(node.labels)
                   
    def get_val(self, p, node=None):
        if node is None:
            node = self.root
            
        if node.labels is not None:
            return self.get_pred(node)
        
        next_node = node.left
        thresh = node.threshold
        
        if p[node.feature_idx] >= thresh:
            next_node = node.right
            
        return self.get_val(p, next_node)
        
    def predict(self, X):
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
        if self.criterion == 'variance' or self.criterion == 'mad_median':
            raise Exception("Invalid Operation for criterion {}".format(self.criterion))
            
        probs = [self.get_probs(p) for p in X]
        return probs 