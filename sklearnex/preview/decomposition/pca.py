#!/usr/bin/env python
# ===============================================================================
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
# ===============================================================================

import numbers
from math import sqrt

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_array, check_is_fitted

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils import _check_array

from ..._device_offload import dispatch
from ..._utils import PatchingConditionsChain

if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
    from sklearn.utils import check_scalar
if sklearn_check_version("0.23"):
    from sklearn.decomposition._pca import _infer_dimension
else:
    from sklearn.decomposition._pca import _infer_dimension_

from sklearn.decomposition import PCA as sklearn_PCA

from onedal.decomposition import PCA as onedal_PCA


class PCA(sklearn_PCA):
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**sklearn_PCA._parameter_constraints}

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

    def _validate_n_components(self, n_components, n_samples, n_features, n_sf_min):
        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if" " n_samples >= n_features"
                )
        elif not 0 <= n_components <= n_sf_min:
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(n_samples, n_features))
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

    def fit(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        elif sklearn_check_version("1.1"):
            check_scalar(
                self.n_oversamples,
                "n_oversamples",
                min_val=1,
                target_type=numbers.Integral,
            )
        self._fit(X)
        return self

    def _fit(self, X):
        if issparse(X):
            raise TypeError(
                "PCA does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )

        if sklearn_check_version("0.23"):
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=False
            )
        else:
            X = _check_array(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=False
            )

        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        if self.n_components is None:
            if self.svd_solver == "arpack":
                n_components = n_sf_min - 1
            else:
                n_components = n_sf_min
        else:
            n_components = self.n_components

        self._validate_n_components(n_components, n_samples, n_features, n_sf_min)

        self._fit_svd_solver = self.svd_solver
        shape_good_for_daal = X.shape[1] / X.shape[0] < 2
        if self._fit_svd_solver == "auto":
            if sklearn_check_version("1.1"):
                if max(X.shape) <= 500 or n_components == "mle":
                    self._fit_svd_solver = "full"
                elif 1 <= n_components < 0.8 * n_sf_min:
                    self._fit_svd_solver = "randomized"
                else:
                    self._fit_svd_solver = "full"
            else:
                if n_components == "mle":
                    self._fit_svd_solver = "full"
                else:
                    n, p, k = X.shape[0], X.shape[1], n_components
                    # check if sklearnex is faster than randomized sklearn
                    # Refer to daal4py
                    regression_coefs = np.array(
                        [
                            [9.779873e-11, n * p * k],
                            [-1.122062e-11, n * p * p],
                            [1.127905e-09, n**2],
                        ]
                    )

                    if (
                        n_components >= 1
                        and np.dot(regression_coefs[:, 0], regression_coefs[:, 1]) <= 0
                    ):
                        self._fit_svd_solver = "randomized"
                    else:
                        self._fit_svd_solver = "full"

        if not shape_good_for_daal or self._fit_svd_solver != "full":
            if sklearn_check_version("0.23"):
                X = self._validate_data(X, copy=self.copy)
            else:
                X = check_array(X, copy=self.copy)

        # Call different fits for either full or truncated SVD
        if shape_good_for_daal and self._fit_svd_solver == "full":
            return dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": sklearn_PCA._fit_full,
                },
                X,
            )
        elif not shape_good_for_daal and self._fit_svd_solver == "full":
            return sklearn_PCA._fit_full(self, X, n_components)
        elif self._fit_svd_solver in ["arpack", "randomized"]:
            return sklearn_PCA._fit_truncated(
                self,
                X,
                n_components,
                self._fit_svd_solver,
            )
        else:
            raise ValueError("Unrecognized svd_solver='{0}'".format(self._fit_svd_solver))

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        if method_name == "fit":
            patching_status = PatchingConditionsChain(
                f"sklearn.decomposition.{class_name}.{method_name}"
            )
            patching_status.and_conditions(
                [
                    (
                        self._fit_svd_solver == "full",
                        f"'{self._fit_svd_solver}' SVD solver is not supported. "
                        "Only 'full' solver is supported.",
                    ),
                ]
            )
            return patching_status
        elif method_name == "transform":
            patching_status = PatchingConditionsChain(
                f"sklearn.decomposition.{class_name}.{method_name}"
            )
            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained"),
                ]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_fit(self, X, y=None, queue=None):
        if self.n_components == "mle" or self.n_components is None:
            onedal_n_components = min(X.shape)
        elif 0 < self.n_components < 1:
            onedal_n_components = min(X.shape)
        else:
            onedal_n_components = self.n_components

        onedal_params = {
            "n_components": onedal_n_components,
            "is_deterministic": True,
            "method": "precomputed",
        }
        self._onedal_estimator = onedal_PCA(**onedal_params)
        self._onedal_estimator.fit(X, queue=queue)
        self._save_attributes()

        U = None
        S = self.singular_values_
        V = self.components_

        return U, S, V

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue)

    def _onedal_transform(self, X):
        X = _check_array(X, dtype=[np.float64, np.float32], ensure_2d=True, copy=False)

        if hasattr(self, "n_features_in_"):
            if self.n_features_in_ != X.shape[1]:
                raise ValueError(
                    f"X has {X.shape[1]} features, "
                    f"but {self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features as input"
                )
        elif hasattr(self, "n_features_"):
            if self.n_features_ != X.shape[1]:
                raise ValueError(
                    f"X has {X.shape[1]} features, "
                    f"but {self.__class__.__name__} is expecting "
                    f"{self.n_features_} features as input"
                )

        # Mean center
        X_centered = X - self.mean_
        return dispatch(
            self,
            "transform",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": sklearn_PCA.transform,
            },
            X_centered,
        )

    def transform(self, X):
        check_is_fitted(self)
        if hasattr(self, "_onedal_estimator"):
            X_new = self._onedal_transform(X)[:, : self.n_components_]
            if self.whiten:
                X_new /= np.sqrt(self.explained_variance_)
        else:
            return sklearn_PCA.transform(self, X)
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
        if self.svd_solver in ["randomized", "arpack"]:
            return sklearn_PCA.fit_transform(self, X)
        else:
            self.fit(X)
            if hasattr(self, "_onedal_estimator"):
                X_new = self._onedal_transform(X)[:, : self.n_components_]
                if self.whiten:
                    X_new /= np.sqrt(self.explained_variance_)
                return X_new
            else:
                return sklearn_PCA.transform(self, X)

    def _save_attributes(self):
        self.n_samples_ = self._onedal_estimator.n_samples_

        if sklearn_check_version("1.2"):
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            n_features = self.n_features_in_
        elif sklearn_check_version("0.24"):
            self.n_features_ = self._onedal_estimator.n_features_
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            n_features = self.n_features_in_
        else:
            self.n_features_ = self._onedal_estimator.n_features_
            n_features = self.n_features_
        n_sf_min = min(self.n_samples_, n_features)

        self.mean_ = self._onedal_estimator.mean_
        self.singular_values_ = self._onedal_estimator.singular_values_
        self.explained_variance_ = self._onedal_estimator.explained_variance_
        self.explained_variance_ratio_ = self._onedal_estimator.explained_variance_ratio_

        if self.n_components is None:
            self.n_components_ = self._onedal_estimator.n_components_
        elif self.n_components == "mle":
            if sklearn_check_version("0.23"):
                self.n_components_ = _infer_dimension(
                    self.explained_variance_, self.n_samples_
                )
            else:
                self.n_components_ = _infer_dimension_(
                    self.explained_variance_, self.n_samples_, n_features
                )
        elif 0 < self.n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            self.n_components_ = (
                np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
            )
        else:
            self.n_components_ = self._onedal_estimator.n_components_

        if self.n_components_ < n_sf_min:
            if self.explained_variance_.shape[0] == n_sf_min:
                self.noise_variance_ = self.explained_variance_[
                    self.n_components_ :
                ].mean()
            else:
                self.noise_variance_ = self._onedal_estimator.noise_variance_
        else:
            self.noise_variance_ = 0.0

        self.explained_variance_ = self.explained_variance_[: self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[
            : self.n_components_
        ]
        self.components_ = self._onedal_estimator.components_[: self.n_components_]
        self.singular_values_ = self.singular_values_[: self.n_components_]
