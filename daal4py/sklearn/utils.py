import numpy as np

def getFPType(X):
    dt = getattr(X, 'dtype', None)
    if dt == np.double:
        return "double"
    elif dt == np.single:
        return "float"
    else:
        raise ValueError("Input array has unexpected dtype = {}".format(dt))

def make2d(X):
    if np.isscalar(X):
        X = np.asarray(X)[np.newaxis, np.newaxis]
    elif isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape((X.size, 1))
    return X

