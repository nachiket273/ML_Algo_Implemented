import numpy as np
from scipy.spatial.distance import cdist

class Kernel():
    def __init__(self):
        pass

    def __call__(self, x, y):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

class Linear(Kernel):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        return np.dot(x, y.T)

    def __repr__(self):
        return "Linear"

class Polynomial(Kernel):
    def __init__(self, degree=3, coef=0, gamma=None):
        super().__init__()
        self.degree = degree
        self.coef = coef
        self.gamma = gamma

    def __call__(self, x, y):
        if self.gamma is None:
            self.gamma = 1/ x.shape[1]
        
        return (self.gamma * np.dot(x, y.T) + self.coef) ** self.degree

    def __repr__(self):
        return "Polynomial"

class RBF(Kernel):
    def __init__(self, gamma= None):
        super().__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gamma is None:
            self.gamma = 1 / x.shape[1]
        return np.exp(-self.gamma * cdist(x, y,  metric='sqeuclidean'))

    def __repr__(self):
        return "RBF"

class Laplace(Kernel):
    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma

    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gamma is None:
            self.gamma = 1/ x.shape[1]
        return np.exp(-self.gamma * cdist(x, y, metric='cityblock'))

    def __repr__(self):
        return "Laplace"

class Sigmoid(Kernel):
    def __init__(self, gamma=None, coef=1):
        super().__init__()
        self.coef = coef
        self.gamma = gamma
    
    def __call__(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.gamma is None:
            self.gamma = 1/ x.shape[1]

        return np.tanh(self.gamma * np.dot(x, y.T) + self.coef)

    def __repr__(self):
        return "Sigmoid Kernel"