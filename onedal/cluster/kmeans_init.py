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

import dpnp
import dpctl.tensor as dpt

from onedal import _backend

# TODO:
# remove unnecessary imports.
from daal4py.sklearn._utils import get_dtype
from ..datatypes import (
    from_table,
    to_table,
    _convert_to_supported)

from ..common._policy import _get_policy

from sklearn.utils import check_random_state

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2023, 'P', 200)):
    class KMeansInit:
        """
        KMeansInit oneDAL implementation.
        """

        def __init__(self,
                     cluster_count,
                     seed=777,
                     local_trials_count=None,
                     algorithm='plus_plus_dense'):
            self.cluster_count = cluster_count
            self.seed = seed
            self.local_trials_count = local_trials_count
            self.algorithm = algorithm

            if local_trials_count is None:
                self.local_trials_count = 2 + int(dpnp.log(cluster_count))
            else:
                self.local_trials_count = local_trials_count

        def _get_policy(self, queue, *data):
            return _get_policy(queue, *data)

        def _get_onedal_params(self, dtype=dpnp.float32):
            return {
                'fptype': 'float' if dtype == dpnp.float32 else 'double',
                'local_trials_count': self.local_trials_count,
                'method': self.algorithm, 'seed': self.seed,
                'cluster_count': self.cluster_count,
            }

        def _get_params_and_input(self, X, policy):
            types = [dpnp.float32, dpnp.float64]
            # TODO:
            # move checking pandas dtypes on sklearnex level.
            # if get_dtype(X_loc) not in types:
            if X.dtype not in types:
                X = X.astype(dpnp.float64)

            # TODO:
            # X = _convert_to_supported(policy, X)

            # TODO:
            # move checking pandas dtypes on sklearnex level.
            # dtype = get_dtype(X_loc)
            dtype = X.dtype
            params = self._get_onedal_params(dtype)
            return (params, to_table(X), dtype)

        def _compute_raw(self, X_table, module, policy, dtype=dpnp.float32):
            params = self._get_onedal_params(dtype)

            result = module.compute(policy, params, X_table)
            # returns onedal table.
            return result.centroids

        def _compute(self, X, module, queue):
            policy = self._get_policy(queue, X)
            _, X_table, dtype = self._get_params_and_input(X, policy)

            centroids = self._compute_raw(X_table, module, policy, dtype)

            # TODO:
            # add from_table.
            return dpnp.array(dpt.asarray(centroids), copy=False)

        def compute_raw(self, X_table, policy, dtype=dpnp.float32):
            return self._compute_raw(X_table, _backend.kmeans_init.init, policy, dtype)

        def compute(self, X, queue=None):
            return self._compute(X, _backend.kmeans_init.init, queue)

    def kmeans_plusplus(
            X,
            n_clusters,
            *,
            x_squared_norms=None,
            random_state=None,
            n_local_trials=None,
            queue=None):
        random_seed = check_random_state(random_state).tomaxint()
        return (
            KMeansInit(
                n_clusters, seed=random_seed, local_trials_count=n_local_trials).compute(
                X, queue), dpnp.full(
                n_clusters, -1))
