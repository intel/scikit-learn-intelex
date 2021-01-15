#===============================================================================
# Copyright 2014-2021 Intel Corporation
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
#===============================================================================

import numpy as np
import numbers
from math import sqrt
from scipy.sparse import issparse

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import stable_cumsum

import daal4py
from .._utils import getFPType, get_patch_message, sklearn_check_version
import logging

if sklearn_check_version('0.22'):
    from sklearn.decomposition._pca import PCA as PCA_original
else:
    from sklearn.decomposition.pca import PCA as PCA_original

if sklearn_check_version('0.23'):
    from sklearn.decomposition._pca import _infer_dimension
elif sklearn_check_version('0.22'):
    from sklearn.decomposition._pca import _infer_dimension_
else:
    from sklearn.decomposition.pca import _infer_dimension_


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

    def _validate_n_components(self, n_components, n_samples, n_features):
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
            if not isinstance(n_components, numbers.Integral):
                raise ValueError("n_components=%r must be of type int "
                                 "when greater than or equal to 1, "
                                 "was of type=%r"
                                 % (n_components, type(n_components)))

    def _fit_full_daal4py(self, X, n_components):
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        if n_components == 'mle':
            daal_n_components = n_features
        elif n_components < 1:
            daal_n_components = n_sf_min
        else:
            daal_n_components = n_components

        fpType = getFPType(X)

        covariance_algo = daal4py.covariance(fptype=fpType, outputMatrixType='covarianceMatrix')
        covariance_res = covariance_algo.compute(X)

        self.mean_ = covariance_res.mean.ravel()
        covariance = covariance_res.covariance
        variances_ = np.array([covariance[i, i] for i in range(n_features)])

        pca_alg = daal4py.pca(
            fptype=fpType,
            method='correlationDense',
            resultsToCompute='eigenvalue',
            isDeterministic=True,
            nComponents=daal_n_components
        )
        pca_res = pca_alg.compute(X, covariance)

        components_ = pca_res.eigenvectors
        explained_variance_ = np.maximum(pca_res.eigenvalues.ravel(), 0)
        tot_var  = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / tot_var

        if n_components == 'mle':
            if sklearn_check_version('0.23'):
                n_components = _infer_dimension(explained_variance_, n_samples)
            else:
                n_components = _infer_dimension_(explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components,
                                           side='right') + 1

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
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = np.sqrt((n_samples - 1) * self.explained_variance_)

    def _fit_full(self, X, n_components):
        X = check_array(X, dtype=[np.float64, np.float32])

        n_samples, n_features = X.shape
        self._validate_n_components(n_components, n_samples, n_features)

        self._fit_full_daal4py(X, min(X.shape))

        U = self._transform_daal4py(X, whiten=True, check_X=False, scale_eigenvalues=True)
        V = self.components_
        S = self.singular_values_

        if n_components == 'mle':
            if sklearn_check_version('0.23'):
                n_components = _infer_dimension(self.explained_variance_, n_samples)
            else:
                n_components = _infer_dimension_(self.explained_variance_, n_samples, n_features)
        elif 0 < n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components,
                                           side='right') + 1

        if n_components < min(n_features, n_samples):
            self.noise_variance_ = self.explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = self.components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = self.explained_variance_[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
        self.singular_values_ = self.singular_values_[:n_components]

        return U, S, V

    def _fit(self, X):
        if issparse(X):
            raise TypeError('PCA does not support sparse input. See '
                            'TruncatedSVD for a possible alternative.')

        if sklearn_check_version('0.23'):
            X = self._validate_data(X, dtype=[np.float64, np.float32],
                                    ensure_2d=True, copy=self.copy)
        else:
            X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True,
                            copy=self.copy)

        if self.n_components is None:
            if self.svd_solver != 'arpack':
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        self._fit_svd_solver = self.svd_solver
        shape_good_for_daal = X.shape[1] / X.shape[0] < 2
        if self._fit_svd_solver == 'auto':
            if max(X.shape) <= 500 or n_components == 'mle':
                self._fit_svd_solver = 'full'
            elif n_components >= 1 and n_components < (.1 if shape_good_for_daal else .8) * min(X.shape):
                self._fit_svd_solver = 'randomized'
            else:
                self._fit_svd_solver = 'full'

        if self._fit_svd_solver == 'full':
            if shape_good_for_daal:
                logging.info("sklearn.decomposition.PCA.fit: " + get_patch_message("daal"))
                result = self._fit_full(X, n_components)
            else:
                logging.info("sklearn.decomposition.PCA.fit: " + get_patch_message("sklearn"))
                result = PCA_original._fit_full(self, X, n_components)
        elif self._fit_svd_solver in ['arpack', 'randomized']:
            logging.info("sklearn.decomposition.PCA.fit: " + get_patch_message("sklearn"))
            result = self._fit_truncated(X, n_components, self._fit_svd_solver)
        else:
            raise ValueError("Unrecognized svd_solver='{0}'"
                             "".format(self._fit_svd_solver))

        return result

    def _transform_daal4py(self, X, whiten=False, scale_eigenvalues=True, check_X=True):
        if sklearn_check_version('0.22'):
            check_is_fitted(self)
        else:
            check_is_fitted(self, ['mean_', 'components_'], all_or_any=all)

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

    def transform(self, X):
        if self.n_components_ > 0:
            logging.info("sklearn.decomposition.PCA.transform: " + get_patch_message("daal"))
            return self._transform_daal4py(X, whiten=self.whiten,
                                           check_X=True, scale_eigenvalues=False)
        else:
            logging.info("sklearn.decomposition.PCA.transform: " + get_patch_message("sklearn"))
            return PCA_original.transform(self, X)

    def fit_transform(self, X, y=None):
        fit_result = self._fit(X)

        if self._fit_svd_solver == 'full':
            if self.n_components_ > 0:
                return self._transform_daal4py(X, whiten=self.whiten, check_X=False, scale_eigenvalues=False)
            return np.empty((self.n_samples_, 0), dtype=X.dtype)
        else:
            U, S, _ = fit_result
            U = U[:, :self.n_components_]

            if self.whiten:
                U *= sqrt(X.shape[0] - 1)
            else:
                U *= S[:self.n_components_]

            return U
