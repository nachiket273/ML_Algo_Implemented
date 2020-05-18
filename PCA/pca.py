import numpy as np
from numpy.linalg import eigh, svd

class pca():
    def __init__(self, n_components=2, solver='svd'):
        self.n_components = n_components
        self.solver = solver.lower()
        assert(self.solver == 'svd' or self.solver == 'eig')
        self.fit_ = False

    def _adjust_signs(self, u, v):
        max_col = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_col, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u,v

    def _calc_conv_mat(self, X):
        mean = np.mean(X, axis=1)
        X_c = X - mean[:, None]
        cov_mat = np.dot(X_c, X_c.T)
        cov_mat = (1 / X.shape[1] -1 ) * cov_mat
        return cov_mat

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        if self.solver == 'svd':
            u, s, vh = svd(X_centered, full_matrices=False)
            _, vh = self._adjust_signs(u, vh)
        else:
            cov_mat = self._calc_conv_mat(X.T)
            s, vh = eigh(cov_mat)
            vh = vh.T
            max_col = np.argmax(np.abs(vh), axis=1)
            signs = np.sign(vh[range(vh.shape[0]), max_col])
            vh *= signs[:, np.newaxis]
            s = -s
            s = np.sqrt(s)

        self.singular_values_ = s[:self.n_components]
        explained_variance_ = s**2 / (X.shape[0] -1)
        explained_variance_ratio_ = explained_variance_ / explained_variance_.sum()
        self.components_ = vh[:self.n_components]
        self.explained_variance_ = explained_variance_[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]
        self.fit_ = True

    def transform(self, X):
        if not self.fit_:
            raise Exception("Run fit fist , before transform.")
        else:
            X_centered = X - self.mean_
            return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)