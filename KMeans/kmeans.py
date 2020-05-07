import math
import numpy as np
from sklearn.base import BaseEstimator

def calc_dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return math.sqrt(np.sum((a-b)**2))

class Kmeans(BaseEstimator):
    """
    Class that implements K-means algorithm.
    
    Args:
        num_clusters        Number of clusters to be formed
                            Default: 8
        init_method         Cluster centroid initilization methods.
                            Currently supports 'random' or 'k-means++'
                            Default: 'k-means++'
        iters               Number of iterations.
                            The centroids are adjusted for number of iterations
                            if centroid change is above tolerance level.
                            Default: 100
        num_seeds           Number of seeds for which the kmeans will be run.
                            The one with best inertia (least squared distance)
                            will be saved.
                            Default: 10
        tolerance           The tolerance value which will be used to decide if
                            kmeans reached minima.
                            Default: 1e-4
        random_seed         Random seed value for numpy operations.
                            This seed will be used as Randomstate to generate
                            number of seeds.
                            Default: 0
    """
    def __init__(self, num_clusters=8, init_method='k-means++', iters=100, 
                    num_seeds=10, tolerance=1e-4, random_seed=0):
        self.num_clusters = num_clusters
        self.init_method = init_method
        self.iters = iters
        self.num_seeds = num_seeds
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.centroids_ = []
        self.clusters_ = None
        self.inertia_ = None

    def _calc_norm(self, X):
        return np.sqrt(np.einsum('ij,ij->i', X, X))

    def _get_tolerance(self, X):
        return np.mean(np.var(X, axis=0)) * self.tolerance

    def _get_next_centroid(self, X):
        dists = np.array([min([calc_dist(x, cent) for cent in self.centroids_]) for x in X])

        # Get squared probabilities
        p = (dists**2) / np.sum((dists**2))
        ind = np.random.choice(X.shape[0], 1, p=p)[0]
        return X[ind]

    def _init_centroid(self, X):
        if isinstance(self.init_method, str):
            if self.init_method == 'random':
                self.centroids_ = np.random.randint(0, X.shape[0], size=(self.num_clusters,))
            elif self.init_method == 'k-means++':
                self.centroids_.append(X[np.random.randint(0, X.shape[0], size=1)[0]])
                for _ in range(1, self.num_clusters):
                    self.centroids_.append(self._get_next_centroid(X))
            else:
               raise ValueError("Invalid value for init method.") 
        else:
            raise ValueError("Invalid value for init method.")

    def _get_closest_centroid(self, x):
        return np.argmin([calc_dist(x, cent) for cent in self.centroids_])

    def _create_cluster(self, X):
        for x in X:
            self.clusters_[self._get_closest_centroid(x)].append(x)

    def _calc_centroids(self):
        self.centroids_ = [np.mean(self.clusters_[i], axis=0) for i in range(self.num_clusters)]

    def _is_minima_reached(self, old_centroids, tol):
        return calc_dist(old_centroids, self.centroids_) <= tol

    def fit(self, X):
        random_state = np.random.RandomState(self.random_seed)
        tolerance = self._get_tolerance(X)
        X_adj = X - np.mean(X)

        # Generate seeds
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.num_seeds)

        best_centroids = []
        best_cluster = []
        best_inertia = np.inf

        for sd in seeds:
            np.random.seed(sd)
            self.centroids_ = []

            # Initialize the clusters
            self._init_centroid(X_adj)

            # Now adjust the centroids
            # Run for number of iterations if tolerance value
            # is not reached.
            for _ in range(self.iters):
                # Assign clusters to samples.
                self.clusters_ = [[] for _ in range(self.num_clusters)]
                self._create_cluster(X_adj)

                # calculate new centroids
                old = self.centroids_
                self._calc_centroids()

                # Check if minima is reached.
                if self._is_minima_reached(old, tolerance):
                    break

            self.inertia_ = np.sum([np.sum((self.clusters_[i]-self.centroids_[i])**2) for i in range(self.num_clusters)])

            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                best_centroids = self.centroids_
                best_cluster = self.clusters_

        self.inertia_ = best_inertia
        self.clusters_ = best_cluster
        self.centroids_ = best_centroids

    def predict(self, X_test):
        X_adj = X_test - np.mean(X_test)
        preds = [self._get_closest_centroid(x) for x in X_adj]
        return preds
