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

from sklearn.utils import check_array

from onedal.datatypes._data_conversion import get_dtype, make2d

from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils._array_api import get_namespace


class BaseDBSCAN(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def _get_onedal_params(self, xp, dtype):
        return {
            "fptype": "float" if dtype == xp.float32 else "double",
            "method": "by_default",
            "min_observations": int(self.min_samples),
            "epsilon": float(self.eps),
            "mem_save_mode": False,
            "result_options": "core_observation_indices|responses",
        }

    def _fit(self, X, xp, is_array_api_compliant, y, sample_weight, queue):
        policy = self._get_policy(queue, X)
        X = check_array(X, accept_sparse="csr", dtype=[xp.float64, xp.float32])
        sample_weight = make2d(sample_weight) if sample_weight is not None else None
        X = make2d(X)

        types = [xp.float32, xp.float64]
        if get_dtype(X) not in types:
            X = X.astype(xp.float64)
        X = _convert_to_supported(policy, X)
        dtype = get_dtype(X)
        params = self._get_onedal_params(xp, dtype)
        result = self._get_backend("dbscan", "clustering", None).compute(
            policy, params, to_table(X), to_table(sample_weight)
        )

        self.labels_ = from_table(result.responses).reshape(-1)
        if result.core_observation_indices is not None:
            self.core_sample_indices_ = from_table(
                result.core_observation_indices
            ).reshape(-1)
        else:
            self.core_sample_indices_ = xp.array([], dtype=xp.int32)
        self.components_ = xp.take(X, self.core_sample_indices_, axis=0)
        self.n_features_in_ = X.shape[1]
        return self


class DBSCAN(BaseDBSCAN):
    def __init__(
        self,
        eps=0.5,
        *,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None, queue=None):
        xp, is_array_api_compliant = get_namespace(X)
        return super()._fit(X, xp, is_array_api_compliant, y, sample_weight, queue)
