#!/usr/bin/env python
#===============================================================================
# Copyright 2023 Intel Corporation
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
from sklearn.utils.extmath import stable_cumsum
from onedal.datatypes import _check_array
from .._device_offload import dispatch
from onedal.decomposition import PCA as onedal_PCA
from sklearn.decomposition import PCA as sklearn_PCA

from sklearn.base import BaseEstimator
from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted
from daal4py.sklearn._utils import sklearn_check_version
if sklearn_check_version('0.23'):
    from sklearn.decomposition._pca import _infer_dimension
elif sklearn_check_version('0.22'):
    from sklearn.decomposition._pca import _infer_dimension_
else:
    from sklearn.decomposition.pca import _infer_dimension_


class PCA(sklearn_PCA):
    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def _fit(self, X):
        if issparse(X):
            raise TypeError(
                "PCA does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )

        X = _check_array(
            X,
            dtype=[np.float64, np.float32],
            ensure_2d=True,
            copy=self.copy
        )
        self.mean_ = np.mean(X, axis=0)
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        if self.n_components is None:
            if self.svd_solver != "arpack":
                n_components = n_sf_min
            else:
                n_components = n_sf_min - 1
        else:
            n_components = self.n_components

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if"
                    " n_samples >= n_features"
                )
        elif not 0 <= n_components <= n_sf_min:
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(n_samples, n_features))
            )

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            elif 1 <= n_components < 0.8 * n_sf_min:
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == "full":
            return dispatch(self, 'decomposition.PCA.fit', {
                'onedal': self.__class__._onedal_fit,
                'sklearn': sklearn_PCA._fit_full,
            }, X)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return sklearn_PCA._fit_truncated(
                self,
                X,
                n_components,
                self._fit_svd_solver,
            )
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'".format(self._fit_svd_solver)
            )

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'decomposition.PCA.fit':
            return self._fit_svd_solver == 'full'
        elif method_name == 'decomposition.PCA.transform':
            return hasattr(self, '_onedal_estimator')
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}'
        )

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'decomposition.PCA.fit':
            return self._fit_svd_solver == 'full'
        elif method_name == 'decomposition.PCA.transform':
            return hasattr(self, '_onedal_estimator')
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}'
        )

    def _onedal_fit(self, X, y=None, queue=None):
        if self._fit_svd_solver == "full":
            method = "precomputed"
        else:
            raise ValueError(
                "Unknown method='{0}'".format(self.svd_solver)
            )

        if self.n_components == 'mle' or self.n_components is None:
            onedal_n_components = min(X.shape)
        elif 0 < self.n_components < 1:
            onedal_n_components = min(X.shape)
        else:
            onedal_n_components = self.n_components

        onedal_params = {
            'n_components': onedal_n_components,
            'is_deterministic': True,
            'method': method,
            'copy': self.copy
        }
        self._onedal_estimator = onedal_PCA(**onedal_params)
        self._onedal_estimator.fit(X, y, queue=queue)
        self._save_attributes()

        U = None
        S = self.singular_values_
        V = self.components_

        return U, S, V

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue)

    def get_precision(self):
        n_features = self.components_.shape[1]

        # handle corner cases first
        if self.n_components_ == 0:
            return np.eye(n_features) / self.noise_variance_

        if np.isclose(self.noise_variance_, 0.0, atol=0.0):
            return np.linalg.inv(self.get_covariance())

        # Get precision using matrix inversion lemma
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.0)
        precision = np.dot(components_, components_.T) / self.noise_variance_
        precision.flat[:: len(precision) + 1] += 1.0 / exp_var_diff
        precision = np.dot(
            components_.T,
            np.dot(np.linalg.inv(precision), components_)
        )
        precision /= -(self.noise_variance_**2)
        precision.flat[:: len(precision) + 1] += 1.0 / self.noise_variance_
        return precision

    def _sklearnex_transform(self, X):
        check_is_fitted(self)
        X = _check_array(
            X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
        )
        # Mean center
        X -= self.mean_
        return dispatch(self, 'decomposition.PCA.transform', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_PCA.transform,
        }, X)

    def transform(self, X):
        if self.svd_solver in ["arpack", "randomized"]:
            return sklearn_PCA.transform(self, X)
        else:
            X_new = self._sklearnex_transform(X)[:, : self.n_components_]
            if self.whiten:
                S_inv = np.diag(1 / self.singular_values_.reshape(-1,))
                X_new = np.sqrt(X.shape[0] - 1) * np.dot(X_new, S_inv)
        return X_new

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values of X.
        """
        U, S, Vt = self._fit(X)
        if U is None:
            X_new = self.transform(X)
        else:
            X_new = U[:, : self.n_components_]
            if self.whiten:
                X_new *= np.sqrt(X.shape[0] - 1)
            else:
                X_new *= S[: self.n_components_]

        return X_new

    def inverse_transform(self, X):
        """Transform data back to its original space.
        In other words, If `X` is the transformed matrix of `X_original`,
        then `X_original` would be returned.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of components.
        Returns
        -------
        X_original array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Notes
        -----
        If whitening is enabled, inverse_transform will compute the
        exact inverse operation, which includes reversing whitening.
        """
        if self.whiten:
            return (
                np.dot(
                    X,
                    np.sqrt(self.explained_variance_[:, np.newaxis])
                    * self.components_
                ) + self.mean_
            )
        else:
            return np.dot(X, self.components_) + self.mean_

    def _save_attributes(self):
        self.n_samples_ = self._onedal_estimator.n_samples_

        if sklearn_check_version("1.2"):
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            n_features = self.n_features_in_
        else:
            self.n_features_ = self._onedal_estimator.n_features_
            n_features = self.n_features_

        self.singular_values_ = np.reshape(
            self._onedal_estimator.singular_values_, (-1, )
        )
        self.explained_variance_ = np.reshape(
            self._onedal_estimator.explained_variance_, (-1, )
        )
        self.explained_variance_ratio_ = np.reshape(
            self._onedal_estimator.explained_variance_ratio_, (-1, )
        )

        if self.n_components is None:
            self.n_components_ = self._onedal_estimator.n_components_
        elif self.n_components == 'mle':
            if sklearn_check_version('0.23'):
                self.n_components_ = _infer_dimension(
                    self.explained_variance_, self.n_samples_
                )
            else:
                self.n_components_ = _infer_dimension_(
                    self.explained_variance_, self.n_samples_, n_features
                )
        elif 0 < self.n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            self.n_components_ = np.searchsorted(
                ratio_cumsum, self.n_components, side='right') + 1
        else:
            self.n_components_ = self._onedal_estimator.n_components_

        self.explained_variance_ = self.explained_variance_[:self.n_components_]
        self.explained_variance_ratio_ = \
            self.explained_variance_ratio_[:self.n_components_]
        self.components_ = \
            self._onedal_estimator.components_[:self.n_components_]
        self.singular_values_ = self.singular_values_[:self.n_components_]
        self.noise_variance_ = self._onedal_estimator.noise_variance_
