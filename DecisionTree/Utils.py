import numpy as np


def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / sum(counts)
    return -1.0 * np.sum(probs * np.log2(probs))

def gini(y):
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / sum(counts)
    return 1- np.sum(probs ** 2)

def variance(y):
    mean = np.sum(y)/y.shape[0]
    return np.sum((y - mean)**2)/y.shape[0]
    
def mad_median(y):
    med = np.median(y)
    return np.sum(y-med)/y.shape[0]