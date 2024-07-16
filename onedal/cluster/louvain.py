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
from sklearn.base import ClusterMixin
from sklearn.utils import check_array

from daal4py.sklearn._utils import get_dtype, make2d

from ..common._base import BaseEstimator
from ..datatypes import from_table, to_graph


class Louvain(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        resolution=1.0,
        *,
        accuracy_threshold=.0001,
        max_iteration_count=10
    ):
        self.resolution = resolution
        self.accuracy_threshold = accuracy_threshold
        self.max_iteration_count = max_iteration_count

    def _get_onedal_params(self, dtype=np.float32):
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "by_default",
            "accuracy_threshold": float(self.accuracy_threshold),
            "resolution": float(self.resolution),
            "max_iteration_count": int(self.max_iteration_count),
        }

    def _fit(self, X, module, queue):
        assert(queue==None)
        X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
        X = make2d(X)

        types = [np.float32, np.float64]
        if get_dtype(X) not in types:
            X = X.astype(np.float64)
        dtype = get_dtype(X)
        params = self._get_onedal_params(dtype)
        result = module.vertex_partioning(params, to_graph(X), to_table(sample_weight))

        self.labels_ = from_table(result.labels).ravel()
        self.modularity_ = float(result.modularity)
        self.community_count = int(result.community_count)
        self.n_features_in_ = X.shape[1]
        return self