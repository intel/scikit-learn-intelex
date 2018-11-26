import numpy as np
from sklearn import decomposition
from sklearn.decomposition.pca import *
from sklearn.decomposition.pca import _infer_dimension_

import daal4py
from ..utils import getFPType


def _daal4py_svd(X):
    X = check_array(X, dtype=[np.float64, np.float32])
    X_fptype = getFPType(X)
    alg = daal4py.svd(
        fptype=X_fptype,
        method='defaultDense',
        leftSingularMatrix='requiredInPackedForm',
        rightSingularMatrix='requiredInPackedForm'
    )
    res = alg.compute(X)
    s = res.singularValues
    U = res.leftSingularMatrix
    V = res.rightSingularMatrix
    return U, np.ravel(s), V


def _fit_full(self, X, n_components):
    """Fit the model by computing full SVD on X"""
    n_samples, n_features = X.shape

    if n_components == 'mle':
        if n_samples < n_features:
            raise ValueError("n_components='mle' is only supported "
                             "if n_samples >= n_features")
    elif not 0 <= n_components <= min(n_samples, n_features):
        raise ValueError("n_components=%r must be between 0 and "
                         "min(n_samples, n_features)=%r with "
                         "svd_solver='full'"
                         % (n_components, min(n_samples, n_features)))
    elif n_components >= 1:
        if not isinstance(n_components, (numbers.Integral, np.integer)):
            raise ValueError("n_components=%r must be of type int "
                             "when greater than or equal to 1, "
                             "was of type=%r"
                             % (n_components, type(n_components)))

    # Center data
    self.mean_ = np.mean(X, axis=0)
    X -= self.mean_

    if X.shape[0] > X.shape[1] and (X.dtype == np.float64 or X.dtype == np.float32):
        U, S, V = _daal4py_svd(X)
    else:
        U, S, V = linalg.svd(X, full_matrices=False)
    # flip eigenvectors' sign to enforce deterministic output
    U, V = svd_flip(U, V)

    components_ = V

    # Get variance explained by singular values
    explained_variance_ = (S ** 2) / (n_samples - 1)
    total_var = explained_variance_.sum()
    explained_variance_ratio_ = explained_variance_ / total_var

    # Postprocess the number of components required
    if n_components == 'mle':
        n_components = \
            _infer_dimension_(explained_variance_, n_samples, n_features)
    elif 0 < n_components < 1.0:
        # number of components for which the cumulated explained
        # variance percentage is superior to the desired threshold
        ratio_cumsum = explained_variance_ratio_.cumsum()
        n_components = np.searchsorted(ratio_cumsum, n_components) + 1

    # Compute noise covariance using Probabilistic PCA model
    # The sigma2 maximum likelihood (cf. eq. 12.46)
    if n_components < min(n_features, n_samples):
        self.noise_variance_ = explained_variance_[n_components:].mean()
    else:
        self.noise_variance_ = 0.

    self.n_samples_, self.n_features_ = n_samples, n_features
    self.components_ = components_[:n_components]
    self.n_components_ = n_components
    self.explained_variance_ = explained_variance_[:n_components]
    self.explained_variance_ratio_ = \
        explained_variance_ratio_[:n_components]
    self.singular_values_ = S[:n_components]

    return U, S, V
