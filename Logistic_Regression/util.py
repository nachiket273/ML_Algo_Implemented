import numpy as np

class CrossEntropy():
    def __init__(self):
        pass

    def loss(self, y, ypreds):
        return -np.sum(np.log(ypreds) * (y)  , axis=1)

    def grad(self, y, ypreds):
        return ypreds - y

class Softmax():
    def __init__(self):
        pass

    def loss(self, x):
        max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - max)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def grad(self, x):
        s_max = self.loss(x)
        return s_max * ( 1- s_max)

class Sigmoid():
    def __init__(self):
        pass

    def loss(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def grad(self, x):
        return self.loss(x)* (1 - self.loss(x))

class BinaryCrossEntropy():
    def __init__(self):
        pass

    def loss(self, y, ypred):
        return  - y * np.log(ypred) - (1-y) * np.log(1-ypred)
    
    def grad(self, y, ypred):
        return ypred - y