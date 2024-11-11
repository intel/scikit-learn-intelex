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

import numpy as np

from daal4py.sklearn._utils import get_dtype, make2d

from .._config import _get_config
from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _check_array
from ..utils._array_api import _asarray, _get_sycl_namespace


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

    def _get_onedal_params(self, dtype=np.float32):
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "by_default",
            "min_observations": int(self.min_samples),
            "epsilon": float(self.eps),
            "mem_save_mode": False,
            "result_options": "core_observation_indices|responses",
        }

    def _fit(self, X, y, sample_weight, module, queue):
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, xp, _ = _get_sycl_namespace(X)

        # All data should use the same sycl queue
        if use_raw_input and sua_iface is not None:
            queue = X.sycl_queue

        policy = self._get_policy(queue, X)

        if not use_raw_input:
            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
            sample_weight = make2d(sample_weight) if sample_weight is not None else None
            X = make2d(X)

        types = [np.float32, np.float64]
        if get_dtype(X) not in types:
            X = X.astype(np.float64)
        X = _convert_to_supported(policy, X)
        dtype = get_dtype(X)
        params = self._get_onedal_params(dtype)

        X_table = to_table(X, sua_iface=sua_iface)
        weights_table = to_table(
            sample_weight, sua_iface=_get_sycl_namespace(sample_weight)[0]
        )

        result = module.compute(policy, params, X_table, weights_table)

        self.labels_ = xp.reshape(
            from_table(result.responses, sua_iface=sua_iface, sycl_queue=queue, xp=xp), -1
        )
        if (
            result.core_observation_indices is not None
            and not result.core_observation_indices.kind == "empty"
        ):
            self.core_sample_indices_ = xp.reshape(
                from_table(
                    result.core_observation_indices,
                    sycl_queue=queue,
                    sua_iface=sua_iface,
                    xp=xp,
                ),
                -1,
            )
        else:
            # TODO:
            # self.core_sample_indices_ = _asarray([], xp, sycl_queue=queue, dtype=xp.int32)
            if sua_iface:
                self.core_sample_indices_ = xp.asarray(
                    [], sycl_queue=queue, dtype=xp.int32
                )
            else:
                self.core_sample_indices_ = xp.asarray([], dtype=xp.int32)
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
        return super()._fit(
            X, y, sample_weight, self._get_backend("dbscan", "clustering", None), queue
        )
