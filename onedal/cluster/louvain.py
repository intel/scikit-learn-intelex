# ===============================================================================
# Copyright 2024 Intel Corporation
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

from daal4py.sklearn._utils import get_dtype

from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import from_table, to_graph, to_table
from ..utils.validation import _check_array, _check_X_y, _is_csr


class Louvain(BaseEstimator, ClusterMixin):

    def __init__(
        self, resolution=1.0, *, accuracy_threshold=0.0001, max_iteration_count=10
    ):
        self.resolution = resolution
        self.accuracy_threshold = accuracy_threshold
        self.max_iteration_count = max_iteration_count

    def _get_onedal_params(self, dtype=np.float64):
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "by_default",
            "accuracy_threshold": float(self.accuracy_threshold),
            "resolution": float(self.resolution),
            "max_iteration_count": int(self.max_iteration_count),
        }

    def fit(self, X, y=None, queue=None):
        # queue is only included to match convention for all onedal estimators
        assert queue is None, "Louvain is implemented only for CPU"
        assert _is_csr(X), "input must be CSR sparse"

        if y is None:
            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        else:
            X, y = _check_X_y(
                X, y, accept_sparse="csr", dtype=[np.float64, np.float32])
            y = y.astype(np.int64) # restriction by oneDAL initial partition

        # limitations in oneDAL's shared object force the topology to double type
        dtype = get_dtype(X)
        params = self._get_onedal_params(dtype)
        X = X.astype(np.float64)

        module = self._get_backend("louvain", "vertex_partitioning", None)

        data = (params, to_graph(X)) if y is None else (params, to_graph(X), to_table(y))
        result = module.vertex_partitioning(*data)

        self.labels_ = from_table(result.labels).ravel()
        self.modularity_ = float(result.modularity)
        self.community_count_ = int(result.community_count)
        self.n_features_in_ = X.shape[1]
        return self
