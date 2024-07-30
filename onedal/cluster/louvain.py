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

import warnings

import numpy as np

from daal4py.sklearn._utils import get_dtype

from ..common._base import BaseEstimator
from ..common._mixin import ClusterMixin
from ..datatypes import from_table, to_graph, to_table
from ..utils.validation import _check_array, _check_X_y, _is_csr


class Louvain(BaseEstimator, ClusterMixin):

    def __init__(self, resolution=1.0, *, tol=0.0001, max_iter=10):
        self.resolution = resolution
        self.tol = tol
        self.max_iter = max_iter

    def _get_onedal_params(self, dtype=np.float64):
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "by_default",
            "resolution": float(self.resolution),
            "accuracy_threshold": float(self.tol),
            "max_iteration_count": int(self.max_iter),
        }

    def fit(self, X, y=None, queue=None):
        # queue is only included to match convention for all onedal estimators
        if queue is not None:
            warnings.warn("Louvain is implemented only for CPU")
        assert _is_csr(X), "input must be CSR sparse"
        assert np.sum(X.diagonal()) == 0.0
        if y is None:
            X = _check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        else:
            X, y = _check_X_y(X, y, accept_sparse="csr", dtype=[np.float64, np.float32])
            y = y.astype(np.int64)  # restriction by oneDAL initial partition

        # limitations in oneDAL's shared object force the topology to double type
        dtype = get_dtype(X)
        params = self._get_onedal_params(dtype)
        X = X.astype(np.float64)

        module = self._get_backend("louvain", "vertex_partitioning", None)

        data = (params, to_graph(X)) if y is None else (params, to_graph(X), to_table(y))
        self.temp_ = data
        self.labels_, self.modularity_, self.community_count_ = None, None, None
        #result = module.vertex_partitioning(*data)
        # check if to_graph is the source source of memory leak (lazily)
        #self.labels_ = from_table(result.labels).ravel()
        #self.modularity_ = float(result.modularity)
        #self.community_count_ = int(result.community_count)
        #self.n_features_in_ = X.shape[1]
        return self
