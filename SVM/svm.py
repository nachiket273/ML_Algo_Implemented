from collections import Counter
import numpy as np
from svm_opt import SVM_Opt as opt

class SVC():
    '''
    Class with SVM binary and multiclass classifier implementation.
    cvxopt quadratic optimization is used to solve optimization problem 
    with soft margins.
    Multi-class classification uses "one-vs-one"(ovo) strategy by default
    and will train num_class * (num_class -1) /2 classifiers.

    Args:
        C                   Penalty term used for soft-margin.
                            Default: 1.0
        kernel              Kernel functions.
                            supported: Linear, Polynomial, RBF, Laplace, Sigmoid
                            Default: linear
        tol                 Tolerance value for langrage multipliers.
                            Default: 1e-5
        iters               Number of iterations used in cvxopt optimization.
                            Default cvxopt value is updated with option 
                            cvxopt.solvers.options['maxiters']
                            Default: 100
        degree              Degree be used for polynomial kernel.
                            Default: 3
        coef                Coeficient value used in polynomial kernel.
                            Default: 0
        gamma               Defines how far influence of training samples.
                            If None, 1/num_features is used.
                            If gamma = 1/ (variance)**2 , the rbf kernel is Gaussian.
                            Default: None
        random_seed         random seed value for numpy operations.
                            Default: 0
    '''
    def __init__(self, C=1.0, kernel='linear', tol=1e-5, iters=100, degree=3, coef =0, gamma=None, random_seed=0):
        self.C = C
        self.kernel = kernel.lower()
        self.tol = tol
        self.iters = iters
        self.degree = degree
        self.coef = coef
        self.gamma = gamma
        self.classifiers = None
        self.random_seed = random_seed
        
    def fit(self, X, y):
        np.random.seed(self.random_seed)
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)
        if self.num_classes_ > 2 :
            self.classifiers = [[] for _ in range(self.num_classes_)]
            for i in range(self.num_classes_):
                classifiers = []
                for j in range(i+1, self.num_classes_):
                    idxs_ = np.where(np.logical_or(y == self.classes_[i], y== self.classes_[j] ))[0]
                    optm = opt(C=self.C, kernel=self.kernel, tol=self.tol, iters=self.iters, degree=self.degree, coef=self.coef, gamma=self.gamma, random_seed=self.random_seed)
                    optm._fit(X[idxs_], y[idxs_])
                    classifiers.append(optm)
                self.classifiers[i] = classifiers
        elif self.num_classes_ == 2:
            optm = opt(C=self.C, kernel=self.kernel, tol=self.tol, iters=self.iters, degree=self.degree, coef=self.coef, gamma=self.gamma)
            optm._fit(X, y)
            self.classifiers = optm
        else:
            raise ValueError("Target should have 2 or more classes.")

    def predict(self, X_test):
        if self.num_classes_ > 2:
            preds = []
            for i in range(self.num_classes_):
                for classifier in self.classifiers[i]:
                    pred = classifier._predict(X_test)
                    preds.append(pred)
            preds = np.array(preds)
            final_preds = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X_test.shape[0])]
            return np.array(final_preds)
        elif self.num_classes_ == 2:
            return self.classifiers._predict(X_test)
        else:
            raise ValueError("Target should have 2 or more classes.")  