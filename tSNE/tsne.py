import math
import numpy as np

def euclidean_dist(x, sqrt=True):
    x = np.array(x, dtype=np.float64)
    x_sq = np.einsum('ij,ij->i', x, x)
    x_sq_l = x_sq[:, np.newaxis]
    x_sq_r = x_sq_l.T
    dist = -2 * np.dot(x, x.T)
    dist += x_sq_l
    dist += x_sq_r
    np.maximum(dist, 0.0)
    np.fill_diagonal(dist, 0.0)
    if sqrt:
        return math.sqrt(dist)
    return dist

EPS = np.finfo(np.double).eps

class tsne():
    def __init__(self, n_components=2, perplexity=30, lr=200, n_iter=1000, prec_tries=100, tol=1e-5, random_seed=0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iter = n_iter
        self.prec_tries = prec_tries
        self.tol = tol
        self.random_seed = random_seed

    def _get_perplexity(self, X_dists):
        entropy = math.log(self.perplexity)
        p = np.zeros((X_dists.shape), dtype=np.float64)

        for i in range(X_dists.shape[0]):
            min_beta, max_beta, beta = -np.inf, np.inf, 1.0
            for _ in range(self.prec_tries):
                sum_neigh_i = 0.0
                for j in range(p.shape[1]):
                    if j != i:
                        p[i, j] = math.exp(-X_dists[i, j] * beta)
                        sum_neigh_i += p[i, j]

                if sum_neigh_i == 0.0 :
                    sum_neigh_i = EPS

                sum_distr = 0.0
                for j in range(p.shape[1]):
                    p[i, j] /= sum_neigh_i
                    sum_distr += X_dists[i, j] * p[i, j]

                new_entropy = math.log(sum_neigh_i) + beta * sum_distr
                diff = new_entropy - entropy

                if np.fabs(diff) < self.tol:
                    break

                if diff > 0.0:
                    min_beta = beta
                    if max_beta == np.inf:
                        beta *= 2.0
                    else:
                        beta = (beta + max_beta)/2.0
                else:
                    max_beta = beta
                    if min_beta == -np.inf:
                        beta /= 2.0
                    else:
                        beta = (beta + min_beta)/2.0

        return p

    def _calc_pairwise_affinities(self, X):
        num_samples = X.shape[0]
        affinities = np.zeros((num_samples, num_samples), dtype=np.float32)
        X_dists = euclidean_dist(X, False)
        affinities = self._get_perplexity(X_dists)
        affinities = (affinities + affinities.T)/2
        affinities_sum = np.maximum(np.sum(affinities), EPS)
        affinities = np.maximum(affinities/affinities_sum, EPS)
        return affinities

    def fit_transform(self, X):
        np.random.seed(self.random_seed)

        # Calculate pairwise affinities
        affinities = self._calc_pairwise_affinities(X)

        # Initialize solution
        Y = 1e-4 * np.random.randn(X.shape[0], self.n_components).astype(np.float32)

        for i in range(self.n_iter):
            dist = euclidean_dist(Y, False)
            dist = 1/ (1.0 + dist)
            t_dist_norm = dist / (2.0 * np.sum(dist))
            t_dist_norm = t_dist_norm.clip(min=EPS)

            grads = np.zeros(Y.shape)
            pqd = (affinities - t_dist_norm) * dist
            for j in range(X.shape[0]):
                grads[j] = np.dot(pqd[j], (Y[j] - Y))

            grads *= 4.0

            Y -= self.lr * grads
        
        return Y