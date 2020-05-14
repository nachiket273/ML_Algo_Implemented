from cvxopt import matrix, solvers
from kernels import Linear, Polynomial, RBF, Laplace, Sigmoid
import numpy as np



class SVM_Opt():
    def __init__(self, C=1.0, kernel='linear', tol=1e-5, iters=100, degree=3, coef =0, gamma=None, random_seed=0):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.iters = iters
        self.degree = degree
        self.coef = coef
        self.gamma = gamma
        self.targets = dict()
        self.random_seed = random_seed

    def _opt(self, X, y, kernels):
        # Use cvxopt to find solution to quadratic optimization
        # https://xavierbourretsicotte.github.io/SVM_implementation.html
        # Implementing the SVM algorithm (Soft Margin)
        num_samples = X.shape[0]
        P = matrix(np.outer(y , y) * kernels, tc='d')
        q = matrix(np.ones((num_samples, 1)) * -1)
        A = matrix(y, (1, num_samples), tc='d')
        b = matrix(np.zeros(1), tc='d')
        G = matrix(np.vstack((-np.identity(num_samples), np.identity(num_samples))))
        h = matrix(np.hstack((np.zeros(num_samples), self.C * np.ones(num_samples))))

        solvers.options['show_progress'] = False

        soln = solvers.qp(P, q, G, h, A, b)
        return soln

    def _get_kernel(self):
        if self.kernel == 'linear':
            return Linear()
        elif self.kernel == 'polynomial':
            return Polynomial(self.degree, self.coef, self.gamma)
        elif self.kernel == 'rbf':
            return RBF(self.gamma)
        elif self.kernel == 'laplace':
            return Laplace(self.gamma)
        else:
            raise ValueError("Invalid kernel provided.")
        
    def _fit(self, X, y):
        np.random.seed(self.random_seed)
        num_samples, num_features = X.shape
        tgts = np.unique(y)
        self.targets[-1] = tgts[0]
        self.targets[1] = tgts[1]
        y = np.where(y == tgts[0], -1, 1)

        if self.gamma is None:
            self.gamma = 1 / num_features
        self.kernel_ = self._get_kernel()

        kernels = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                kernels[i, j] = self.kernel_(X[i], X[j])

        soln = self._opt(X, y, kernels)
        mult = np.ravel(soln['x'])

        # Extract support vectors
        idx = mult > self.tol
        self.support_ = np.arange(len(mult))[idx]
        self.lagr_mults_ = mult[idx]

        # support vectors
        self.support_vectors_ = X[idx]
        # labels
        self.labels_ = y[idx]
        self.n_support_ = [len(self.labels_[self.labels_ == -1]), len(self.labels_[self.labels_ == 1])]

        self.dual_coef_ = self.lagr_mults_ * self.labels_

        if self.kernel == 'linear':
            self.coef_ = np.dot(self.dual_coef_, self.support_vectors_)
        else:
            self.coef_ = None

        self.intercept_ = self.labels_[0]
        for i in range(len(self.lagr_mults_)):
            self.intercept_ -= self.dual_coef_[i] * kernels[self.support_[i], 0]

    def _predict(self, X_test):
        if self.coef_ is not None:
            preds = np.dot(X_test, self.coef_) + self.intercept_
        else:
            preds = np.dot(self.dual_coef_, self.kernel_(self.support_vectors_ , X_test)) + self.intercept_

        preds = np.where(preds >= 0, self.targets[1], self.targets[-1])
        return preds
