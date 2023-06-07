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

from onedal import _backend

from daal4py.sklearn._utils import get_dtype
from ..datatypes import _convert_to_supported

from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted
from ..datatypes._data_conversion import from_table, to_table

class KMeansInit:
    """
    KMeansInit oneDAL implementation.
    """
    def __init__(self,
                 cluster_count,
                 seed = 777,
                 local_trials_count = -1,
                 algorithm='plus_plus_dense'):
        self.cluster_count = cluster_count
        self.seed = seed
        self.local_trials_count = local_trials_count
        self.algorithm = algorithm

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_onedal_params(self, dtype=np.float32):
        return {
            'fptype': 'float' if dtype == np.float32 else 'double',
            'local_trials_count': self.local_trials_count,
            'method': self.algorithm, 'seed': self.seed,
            'cluster_count': self.cluster_count,
        }

    def _compute(self, X, module, queue):
        policy = self._get_policy(queue, X)

        X_loc = np.asarray(X)
        dtype = get_dtype(X_loc)
        if dtype not in [np.float32, np.float64]:
            dtype = np.float64
            X_loc = X_loc.astype(dtype)

        X_loc = _convert_to_supported(policy, X_loc)
        params = self._get_onedal_params(get_dtype(X_loc))

        X_table= to_table(X_loc)

        result = module.compute(policy, params, X_table)

        return from_table(result.centroids)

    def compute(self, X, queue = None):
        return self._compute(X, _backend.kmeans_init.init, queue = queue)

def kmeans_plusplus(X, n_clusters, *, x_squared_norms=None, random_state=None, n_local_trials=None, queue=None):
    random_state = 777 if random_state is None else random_state
    n_local_trials = (2 + int(np.log(n_clusters))) if n_local_trials is None else n_local_trials
    return (KMeansInit(n_clusters, random_state, n_local_trials).compute(X, queue), [-1] * n_clusters)
