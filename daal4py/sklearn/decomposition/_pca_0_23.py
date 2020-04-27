#
#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

import numpy as np
import numbers

from sklearn import decomposition
from sklearn.utils import check_array

from sklearn.decomposition._pca import PCA as PCA_original
from sklearn.decomposition._pca import (_infer_dimension, svd_flip)

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import stable_cumsum
from scipy.sparse import issparse

import daal4py
from .._utils import getFPType, method_uses_sklearn, method_uses_daal
import logging


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


def _validate_n_components(n_components, n_samples, n_features):
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


def _process_n_components_None(self_n_components, self_svd_solver, X_shape):
    # Handle n_components==None
    if self_n_components is None:
        if self_svd_solver != 'arpack':
            n_components = min(X_shape)
        else:
            n_components = min(X_shape) - 1
    else:
        n_components = self_n_components

    return n_components


def _n_components_from_fraction(explained_variance_ratio, frac):
    # number of components for which the cumulated explained
    # variance percentage is superior to the desired threshold
    # side='right' ensures that number of features selected
    # their variance is always greater than n_components float
    # passed. More discussion in issue: #15669
    ratio_cumsum = stable_cumsum(explained_variance_ratio)
    n_components = np.searchsorted(ratio_cumsum, frac,
                                   side='right') + 1
    return n_components
    

def _fit_full(self, X, n_components):
    """Fit the model by computing full SVD on X"""
    n_samples, n_features = X.shape

    _validate_n_components(n_components, n_samples, n_features)

    # Center data
    self.mean_ = np.mean(X, axis=0)
    X -= self.mean_

    if X.shape[0] > X.shape[1] and (X.dtype == np.float64 or X.dtype == np.float32):
        U, S, V = _daal4py_svd(X)
    else:
        U, S, V = np.linalg.svd(X, full_matrices=False)
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
            _infer_dimension(explained_variance_, n_samples, n_features)
    elif 0 < n_components < 1.0:
        n_components = _n_components_from_fraction(
            explained_variance_ratio_, n_components)

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


_fit_full_copy = _fit_full

class PCA_prev(PCA_original):
    __doc__ = PCA_original.__doc__

    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def _fit_full(self, X, n_components):
        return _fit_full_copy(self, X, n_components)


class PCA(PCA_original):
    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state


    def _fit_daal4py(self, X, n_components):
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        _validate_n_components(n_components, n_samples, n_features)

        if n_components == 'mle':
            daal_n_components = n_features
        elif n_components < 1:
            daal_n_components = n_sf_min
        else:
            daal_n_components = n_components

        fpType = getFPType(X)
        centering_algo = daal4py.normalization_zscore(
            fptype=fpType, doScale=False)
        pca_alg = daal4py.pca(
            fptype=fpType,
            method='svdDense',
            normalization=centering_algo,
            resultsToCompute='mean|variance|eigenvalue',
            isDeterministic=True,
            nComponents=daal_n_components
        )
        pca_res = pca_alg.compute(X)

        self.mean_ = pca_res.means.ravel()
        variances_ = pca_res.variances.ravel()
        components_ = pca_res.eigenvectors
        explained_variance_ = pca_res.eigenvalues.ravel()
        tot_var  = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / tot_var

        if n_components == 'mle':
            n_components = \
                _infer_dimension(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            n_components = _n_components_from_fraction(
                explained_variance_ratio_, n_components)

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < n_sf_min:
            if explained_variance_.shape[0] == n_sf_min:
                self.noise_variance_ = explained_variance_[n_components:].mean()
            else:
                resid_var_ = variances_.sum()
                resid_var_ -= explained_variance_[:n_components].sum()
                self.noise_variance_ = resid_var_ / (n_sf_min - n_components)
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            explained_variance_ratio_[:n_components]
        self.singular_values_ = np.sqrt((n_samples - 1) * self.explained_variance_)


    def _transform_daal4py(self, X, whiten=False, scale_eigenvalues=True, check_X=True):
        check_is_fitted(self)

        X = check_array(X, dtype=[np.float64, np.float32], force_all_finite=check_X)
        fpType = getFPType(X)

        tr_data = dict()
        if self.mean_ is not None:
            tr_data['mean'] = self.mean_.reshape((1, -1))
        if whiten:
            if scale_eigenvalues:
                tr_data['eigenvalue'] = (self.n_samples_ - 1) * self.explained_variance_.reshape((1, -1))
            else:
                tr_data['eigenvalue'] = self.explained_variance_.reshape((1, -1))
        elif scale_eigenvalues:
            tr_data['eigenvalue'] = np.full(
                (1, self.explained_variance_.shape[0]),
                self.n_samples_ - 1.0, dtype=X.dtype)

        if X.shape[1] != self.n_features_:
            raise ValueError("The number of features of the input data, {}, is not "
                              "equal to the number of features of the training data, {}".format(
                                  X.shape[1], self.n_features_))
        tr_res = daal4py.pca_transform(
            fptype=fpType
        ).compute(X, self.components_, tr_data)

        return tr_res.transformedData


    def _fit_full_daal4py(self, X, n_components):
        n_samples, n_features = X.shape

        # due to need to flip components, need to do full decomposition
        self._fit_daal4py(X, min(n_samples, n_features))
        U = self._transform_daal4py(X, whiten=True, check_X=False, scale_eigenvalues=True)
        V = self.components_
        U, V = svd_flip(U, V)
        U = U.copy()
        V = V.copy()
        S = self.singular_values_.copy()

        if n_components == 'mle':
            n_components = \
                _infer_dimension(self.explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            n_components = _n_components_from_fraction(
                self.explained_variance_ratio_, n_components)

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = self.explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = self.components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = \
            self.explained_variance_ratio_[:n_components]
        self.singular_values_ = self.singular_values_[:n_components]

        return U, S, V


    def _fit_full_vanilla(self, X, n_components):
        """Fit the model by computing full SVD on X"""
        n_samples, n_features = X.shape

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, V = np.linalg.svd(X, full_matrices=False)
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
               _infer_dimension(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            n_components = _n_components_from_fraction(
                explained_variance_ratio_, n_components)

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


    def _fit_full(self, X, n_components):
        n_samples, n_features = X.shape

        _validate_n_components(n_components, n_samples, n_features)

        if n_samples > n_features and (X.dtype == np.float64 or X.dtype == np.float32):
            logging.info("sklearn.decomposition.PCA.fit: " + method_uses_daal)
            return self._fit_full_daal4py(X, n_components)
        else:
            logging.info("sklearn.decomposition.PCA.fit: " + method_uses_sklearn)
            return self._fit_full_vanilla(X, n_components)


    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                        copy=self.copy)

        # Handle n_components==None
        n_components = _process_n_components_None(
            self.n_components, self.svd_solver, X.shape)

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == 'auto':
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == 'mle':
                self._fit_svd_solver = 'full'
            elif n_components >= 1 and n_components < .8 * min(X.shape):
                self._fit_svd_solver = 'randomized'
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = 'full'

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == 'full':
            return self._fit_full(X, n_components)
        elif self._fit_svd_solver in ['arpack', 'randomized']:
            logging.info("sklearn.decomposition.PCA.fit: " + method_uses_sklearn)
            return self._fit_truncated(X, n_components, self._fit_svd_solver)
        elif self._fit_svd_solver == 'daal':
            if X.shape[0] < X.shape[1]:
                raise ValueError("svd_solver='daal' is applicable for tall and skinny inputs only.")
            logging.info("sklearn.decomposition.PCA.fit: " + method_uses_daal)
            return self._fit_daal4py(X, n_components)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'"
                             "".format(self._fit_svd_solver))


    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        if (self.svd_solver == 'daal' and isinstance(X, np.ndarray) and
               X.shape[0] >= X.shape[1]):
            # Handle n_components==None
            n_components = _process_n_components_None(
                self.n_components, self.svd_solver, X.shape)
            logging.info("sklearn.decomposition.PCA.fit: " + method_uses_daal)
            self._fit_daal4py(X, n_components)
            logging.info("sklearn.decomposition.PCA.transform: " + method_uses_daal)
            if self.n_components_ > 0:
                return self._transform_daal4py(X, whiten=self.whiten, check_X=False)
            else:
                return np.empty((self.n_samples_, 0), dtype=X.dtype)
        else:
            U, S, V = self._fit(X)
            U = U[:, :self.n_components_]

            logging.info("sklearn.decomposition.PCA.transform: " + method_uses_sklearn)
            if self.whiten:
                # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
                U *= np.sqrt(X.shape[0] - 1)
            else:
                # X_new = X * V = U * S * V^T * V = U * S
                U *= S[:self.n_components_]

            return U

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        Examples
        --------

        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, copy=True, n_components=2, whiten=False)
        >>> ipca.transform(X) # doctest: +SKIP
        """
        check_is_fitted(self)

        X = check_array(X)
        if self.n_components_ > 0:
            logging.info("sklearn.decomposition.PCA.transform: " + method_uses_daal)
            return self._transform_daal4py(X, whiten=self.whiten,
                                           check_X=False, scale_eigenvalues=False)
        else:
            logging.info("sklearn.decomposition.PCA.transform: " + method_uses_sklearn)
            if self.mean_ is not None:
                X = X - self.mean_
                X_transformed = np.dot(X, self.components_.T)
            if self.whiten:
                X_transformed /= np.sqrt(self.explained_variance_)
            return X_transformed

if (lambda s: (int(s[:4]), int(s[6:])))( daal4py.__daal_link_version__[:8] ) < (2019, 4):
    # with DAAL < 2019.4 PCA only optimizes fit, using DAAL's SVD
    class PCA(PCA_original):
        __doc__ = PCA_original.__doc__

        def __init__(self, n_components=None, copy=True, whiten=False,
                     svd_solver='auto', tol=0.0, iterated_power='auto',
                     random_state=None):
            self.n_components = n_components
            self.copy = copy
            self.whiten = whiten
            self.svd_solver = svd_solver
            self.tol = tol
            self.iterated_power = iterated_power
            self.random_state = random_state

        def _fit_full(self, X, n_components):
            return _fit_full_copy(self, X, n_components)
