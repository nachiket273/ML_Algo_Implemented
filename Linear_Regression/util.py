import numpy as np

def normalize(x, scale=True):
    offset = np.average(x, axis=0)
    x = x - offset
    if scale:
        x = x.T
        norms = np.sqrt(np.einsum('ij,ij->i', x, x))
        norms[norms == 0.0] = 1.0
        x /= norms[:, np.newaxis]
        x = x.T
        return x, offset, norms
    return x, offset

def ols(X, y):
    # Ordinary Least square solution to wX  = y
    # w = (X_transpose * X)_inverse * X_transpose * y
    # Keeping parity with scipy's lstsq 
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html)
    # this function also returns
    # square 2 - norm (y-wx) for solution , degree of freedom and Singular values of
    # X.
    X_T = np.transpose(X)
    inv = np.linalg.inv(np.dot(X_T, X))
    coef = np.dot(np.dot(inv, X_T), y)
    rank = np.linalg.matrix_rank(X)
    res = np.sum(np.abs(y - np.dot(X, coef))**2)
    sv = np.linalg.svd(X)[1]
    return coef, res, rank, sv