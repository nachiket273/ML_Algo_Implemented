import numpy as np


def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    dist = 0

    for i, j in zip(a, b):
        dist += (i-j)**2

    return np.sqrt(dist)

class DBSCAN():
    """
    Class with simple implementation of DBSCAN.

    Args:
        min_samples Minimum number of neighbors to be needed within distance
                    less than eps.
                    Default: 5

        eps         Maximum distance criteria for core samples in cluster.
                    Default: 0.5
    """
    def __init__(self, min_samples=5, eps=0.5):
        self.min_samples = min_samples
        self.eps = eps
        self.neighbors_ = None
        self.clusters_ = []
        self.visited_ = None

    def find_neighbors(self, X):
        self.neighbors_ = [[] for _ in range(X.shape[0])]
        for i, x in enumerate(X):
            for j, y in enumerate(X):
                if i != j and dist(x, y) < self.eps:
                    self.neighbors_[i].append(j)

    def create_cluster(self, X, i):
        clust = [i]

        for neighbor in self.neighbors_[i]:
            if not self.visited_[neighbor]:
                self.visited_[neighbor] = True
                if len(self.neighbors_[neighbor]) >= self.min_samples:
                    extended_clust = self.create_cluster(X, neighbor)
                    clust = clust + extended_clust
                else:
                    clust.append(neighbor)

        return clust

    def get_labels(self, X):
        labels = np.full(shape=X.shape[0], fill_value=len(self.clusters_))
        for i , cluster in enumerate(self.clusters_):
            for item in cluster:
                labels[item] = i
        self.labels_ = labels

    def fit(self, X):
        self.visited_ = [False for _ in range(X.shape[0])]
        self.find_neighbors(X)

        for i in range(X.shape[0]):
            if not self.visited_[i]:
                self.visited_[i] = True

                if len(self.neighbors_[i]) >= self.min_samples:
                    cluster = self.create_cluster(X, i)
                    self.clusters_.append(cluster)

        
        self.get_labels(X)
