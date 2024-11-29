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
from scipy.sparse import issparse
from sklearn.utils import check_random_state

from daal4py.sklearn._utils import daal_check_version, get_dtype
from onedal.common._backend import bind_default_backend

from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _check_array

if daal_check_version((2023, "P", 200)):

    class KMeansInit:
        """
        KMeansInit oneDAL implementation.
        """

        def __init__(
            self,
            cluster_count,
            seed=777,
            local_trials_count=None,
            algorithm="plus_plus_dense",
        ):
            self.cluster_count = cluster_count
            self.seed = seed
            self.local_trials_count = local_trials_count
            self.algorithm = algorithm

            if local_trials_count is None:
                self.local_trials_count = 2 + int(np.log(cluster_count))
            else:
                self.local_trials_count = local_trials_count

        @bind_default_backend("kmeans_init.init", lookup_name="compute")
        def backend_compute(self, params, X_table, queue=None): ...

        def _get_onedal_params(self, dtype=np.float32):
            return {
                "fptype": dtype,
                "local_trials_count": self.local_trials_count,
                "method": self.algorithm,
                "seed": self.seed,
                "cluster_count": self.cluster_count,
            }

        def _get_params_and_input(self, X):
            X = _check_array(
                X,
                dtype=[np.float64, np.float32],
                accept_sparse="csr",
                force_all_finite=False,
            )

            X = _convert_to_supported(X)

            dtype = get_dtype(X)
            params = self._get_onedal_params(dtype)
            return (params, to_table(X), dtype)

        def compute(self, X, queue=None):
            _, X_table, dtype = self._get_params_and_input(X)

            centroids = self.compute_raw(X_table, dtype, queue=queue)

            return from_table(centroids)

        def compute_raw(self, X_table, dtype=np.float32, queue=None):
            params = self._get_onedal_params(dtype)
            result = self.backend_compute(params, X_table, queue=queue)
            return result.centroids

    def kmeans_plusplus(
        X,
        n_clusters,
        *,
        x_squared_norms=None,
        random_state=None,
        n_local_trials=None,
        queue=None,
    ):
        random_seed = check_random_state(random_state).tomaxint()
        return (
            KMeansInit(
                n_clusters, seed=random_seed, local_trials_count=n_local_trials
            ).compute(X, queue),
            np.full(n_clusters, -1),
        )
