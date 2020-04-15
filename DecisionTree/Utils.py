import numpy as np


#https://scikit-learn.org/stable/modules/tree.html?highlight=gini
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / np.sum(counts, axis=0, keepdims=True)
    return -1.0 * np.sum(probs * np.log2(probs))

def gini(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / np.sum(counts, axis=0, keepdims=True)
    return np.sum(probs * (1-probs))

def mse(y):
    mean = np.mean(y, axis=0)
    return np.average((y - mean) ** 2, axis=0)
    
def mae(y):
    mean = np.mean(y, axis=0)
    return np.average(np.abs(y - mean), axis=0)