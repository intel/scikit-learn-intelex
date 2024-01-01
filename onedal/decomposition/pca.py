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
from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils.extmath import stable_cumsum

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes import _convert_to_supported, from_table, to_table

# from abc import ABCMeta, abstractmethod
# from sklearn.decomposition._base import _BasePCA



# class BasePCA(_BasePCA, metaclass=ABCMeta):


class PCA:
    def __init__(
        self,
        n_components=None,
        is_deterministic=True,
        method="cov",
        whiten=False,
    ):
        self.n_components = n_components
        self.method = method
        self.is_deterministic = is_deterministic
        self.whiten = whiten

    def get_onedal_params(self, data, stage="train"):
        if stage == "train":
            n_components = self._resolve_n_components_for_training(data.shape)
        else:
            n_components = self._resolve_n_components_for_result(data.shape)
        return {
            "fptype": "float" if data.dtype == np.float32 else "double",
            "method": self.method,
            "n_components": n_components,
            "is_deterministic": self.is_deterministic,
            "whiten": self.whiten,
        }

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _resolve_n_components_for_training(self, shape_tuple):
        if self.n_components is None or self.n_components == "mle":
            return min(shape_tuple)
        elif 0 < self.n_components < 1:
            return min(shape_tuple)
        else:
            return self.n_components

    def _resolve_n_components_for_result(self, shape_tuple):
        if self.n_components is None:
            return min(shape_tuple)
        elif self.n_components == "mle":
            return _infer_dimension(self.explained_variance_, shape_tuple[0])
        elif 0 < self.n_components < 1:
            ratio_cumsum = stable_cumsum(self.explained_variance_ratio_)
            return np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1
        else:
            return self.n_components

    def _compute_noise_variance(self, n_components, n_sf_min):
        if n_components < n_sf_min:
            if len(self.explained_variance_) == n_sf_min:
                return self.explained_variance_[n_components:].mean()
            elif len(self.explained_variance_) < n_sf_min:
                resid_var = self.variances_.sum()
                resid_var -= self.explained_variance_.sum()
                return resid_var / (n_sf_min - n_components)
        else:
            return 0.0

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
        pca_result = _backend.decomposition.dim_reduction.train(
            policy, params, to_table(X)
        )

        self.mean_ = from_table(pca_result.means).ravel()
        self.variances_ = from_table(pca_result.variances)
        self.components_ = from_table(pca_result.eigenvectors)
        self.singular_values_ = from_table(pca_result.singular_values).ravel()
        self.explained_variance_ = np.maximum(
            from_table(pca_result.eigenvalues).ravel(), 0
        )
        self.explained_variance_ratio_ = from_table(pca_result.explained_variances_ratio)
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        n_components = self._resolve_n_components_for_result(X.shape)
        self.n_components_ = n_components
        self.noise_variance_ = self._compute_noise_variance(n_components, n_sf_min)

        if n_components < params["n_components"]:
            self.explained_variance_ = self.explained_variance_[:n_components]
            self.components_ = self.components_[:n_components]
            self.singular_values_ = self.singular_values_[:n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:n_components]
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
        params = self.get_onedal_params(X, stage="predict")
        # print("n_componets:", params["n_components"], "eigen_vector:", self.components_.shape[0])
        # assert params["n_components"] == self.components_.shape[0]
        result = _backend.decomposition.dim_reduction.infer(
            policy, params, model, to_table(X)
        )
        return from_table(result.transformed_data)
