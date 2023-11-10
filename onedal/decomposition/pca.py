# ==============================================================================
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
# ==============================================================================

import numpy as np

from daal4py.sklearn._utils import sklearn_check_version
from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes import _convert_to_supported, from_table, to_table
from sklearn.utils.extmath import stable_cumsum

if sklearn_check_version("0.23"):
    from sklearn.decomposition._pca import _infer_dimension
else:
    from sklearn.decomposition._pca import _infer_dimension_


class PCA:
    def __init__(
        self,
        n_components=None,
        is_deterministic=True,
        method="precomputed",
        copy=True,
        whiten=False,
    ):
        self.n_components = n_components
        self.method = method
        self.is_deterministic = is_deterministic
        self.whiten = whiten

    def get_onedal_params(self, data):
        n_components = self._resolve_n_components(data.shape)
        return {
            "fptype": "float" if data.dtype == np.float32 else "double",
            "method": self.method,
            "n_components": n_components,
            "is_deterministic": self.is_deterministic,
        }

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _resolve_n_components(self, shape_tuple):
        if self.n_components is None or self.n_components == "mle":
            return min(shape_tuple)
        elif 0 < self.n_components < 1:
            return min(shape_tuple)
        else:
            return self.n_components

    def fit(self, X, queue):
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        policy = self._get_policy(queue, X)
        # TODO: investigate why np.ndarray with OWNDATA=FALSE flag
        # fails to be converted to oneDAL table
        if isinstance(X, np.ndarray) and not X.flags["OWNDATA"]:
            X = X.copy()
        X = _convert_to_supported(policy, X)

        params = self.get_onedal_params(X)
        cov_result = _backend.covariance.compute(
            policy, {"fptype": params["fptype"], "method": "dense"}, to_table(X)
        )
        covariance_matrix = from_table(cov_result.cov_matrix)
        self.mean_ = from_table(cov_result.means)
        pca_result = _backend.decomposition.dim_reduction.train(
            policy, params, to_table(covariance_matrix)
        )

        self.variances_ = from_table(pca_result.variances)
        self.components_ = from_table(pca_result.eigenvectors)
        self.explained_variance_ = np.maximum(
            from_table(pca_result.eigenvalues).ravel(), 0
        )
        total_variance = covariance_matrix.trace()
        self.singular_values_ = np.sqrt((n_samples - 1) * self.explained_variance_)
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        if self.n_components is None:
            self.n_components_ = params["n_components"]
        elif self.n_components == "mle":
            if sklearn_check_version("0.23"):
                self.n_components_ = _infer_dimension(
                    self.explained_variance_, n_samples
                )
            else:
                self.n_components_ = _infer_dimension_(
                    self.explained_variance_, n_samples, n_features
                )
        elif 0 < self.n_components < 1.0:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            self.n_components_ = (
                np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
            )
        else:
            self.n_components_ = params["n_components"]

        if self.n_components_ < n_sf_min:
            if self.explained_variance_.shape[0] == n_sf_min:
                self.noise_variance_ = self.explained_variance_[self.n_components_ :].mean()
            elif self.explained_variance_.shape[0] < n_sf_min:
                resid_var = total_variance - self.explained_variance_[: self.n_components_].sum()
                self.noise_variance_ = resid_var / (n_sf_min - self.n_components_)
        else:
            self.noise_variance_ = 0.0

        self.explained_variance_ = self.explained_variance_[: self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.components_ = self.components_[: self.n_components_]
        self.singular_values_ = self.singular_values_[: self.n_components_]
        return self

    def _create_model(self):
        m = _backend.decomposition.dim_reduction.model()
        m.eigenvectors = to_table(self.components_)
        m.means = to_table(self.mean_)
        if self.whiten:
            m.eigenvalues = to_table(self.explained_variance_)
        self._onedal_model = m
        return m

    def predict(self, X, queue):
        policy = self._get_policy(queue, X)
        model = self._create_model()

        X = _convert_to_supported(policy, X)
        params = self.get_onedal_params(X)
        result = _backend.decomposition.dim_reduction.infer(
            policy, params, model, to_table(X)
        )
        return from_table(result.transformed_data)
