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

from onedal import _backend
from ..common._policy import _get_policy
from ..datatypes._data_conversion import from_table, to_table
from ..datatypes import _convert_to_supported
from daal4py.sklearn._utils import sklearn_check_version


class PCA():
    def __init__(
        self,
        n_components=None,
        is_deterministic=True,
        method='precomputed',
        copy=True
    ):
        self.n_components = n_components
        self.method = method
        self.is_deterministic = is_deterministic

    def get_onedal_params(self, data):
        return {
            'fptype':
                'float' if data.dtype == np.float32 else 'double',
            'method': self.method,
            'n_components': self.n_components,
            'is_deterministic': self.is_deterministic
        }

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def fit(self, X, queue):
        n_samples, n_features = X.shape
        n_sf_min = min(n_samples, n_features)

        policy = self._get_policy(queue, X)
        # TODO: investigate why np.ndarray with OWNDATA=FALSE flag
        # fails to be converted to oneDAL table
        if isinstance(X, np.ndarray) and not X.flags['OWNDATA']:
            X = X.copy()
        X = _convert_to_supported(policy, X)

        params = self.get_onedal_params(X)
        cov_result = _backend.covariance.compute(
            policy,
            {'fptype': params['fptype'], 'method': 'dense'},
            to_table(X)
        )
        covariance_matrix = from_table(cov_result.cov_matrix)
        self.mean_ = from_table(cov_result.means)
        result = _backend.decomposition.dim_reduction.train(
            policy,
            params,
            to_table(covariance_matrix)
        )

        self.n_components_ = self.n_components
        self.variances_ = from_table(result.variances)
        self.components_ = from_table(result.eigenvectors)
        self.explained_variance_ = \
            np.maximum(from_table(result.eigenvalues).ravel(), 0)
        tot_var = covariance_matrix.trace()
        self.explained_variance_ratio_ = self.explained_variance_ / tot_var
        self.singular_values_ = np.sqrt(
            (n_samples - 1) * self.explained_variance_
        )

        if sklearn_check_version("1.2"):
            self.n_features_in_ = n_features
        elif sklearn_check_version("0.24"):
            self.n_features_in_ = n_features
            self.n_features_ = n_features
        else:
            self.n_features_ = n_features

        self.n_samples_ = n_samples
        if self.n_components < n_sf_min:
            if self.explained_variance_.shape[0] < n_sf_min:
                resid_var_ = tot_var - \
                    self.explained_variance_[:self.n_components].sum()
                self.noise_variance_ = \
                    resid_var_ / (n_sf_min - self.n_components)
        return self

    def _create_model(self):
        m = _backend.decomposition.dim_reduction.model()
        m.eigenvectors = to_table(self.components_)
        self._onedal_model = m
        return m

    def predict(self, X, queue):
        policy = self._get_policy(queue, X)
        model = self._create_model()

        X = _convert_to_supported(policy, X)
        params = self.get_onedal_params(X)
        result = _backend.decomposition.dim_reduction.infer(policy,
                                                            params,
                                                            model,
                                                            to_table(X))
        return from_table(result.transformed_data)
