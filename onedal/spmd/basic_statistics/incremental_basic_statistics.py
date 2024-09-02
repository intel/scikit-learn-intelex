# ==============================================================================
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
# ==============================================================================

from daal4py.sklearn._utils import get_dtype

from ...basic_statistics import (
    IncrementalBasicStatistics as base_IncrementalBasicStatistics,
)
from ...datatypes import _convert_to_supported, to_table
from .._base import BaseEstimatorSPMD


class IncrementalBasicStatistics(BaseEstimatorSPMD, base_IncrementalBasicStatistics):
    def _reset(self):
        self._partial_result = super(base_IncrementalBasicStatistics, self)._get_backend(
            "basic_statistics", None, "partial_compute_result"
        )

    def partial_fit(self, X, weights=None, queue=None):
        """
        Computes partial data for basic statistics
        from data batch X and saves it to `_partial_result`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data batch, where `n_samples` is the number of samples
            in the batch, and `n_features` is the number of features.

        queue : dpctl.SyclQueue
            If not None, use this queue for computations.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._queue = queue
        policy = super(base_IncrementalBasicStatistics, self)._get_policy(queue, X)
        X, weights = _convert_to_supported(policy, X, weights)

        if not hasattr(self, "_onedal_params"):
            dtype = get_dtype(X)
            self._onedal_params = self._get_onedal_params(False, dtype=dtype)

        X_table, weights_table = to_table(X, weights)
        self._partial_result = super(base_IncrementalBasicStatistics, self)._get_backend(
            "basic_statistics",
            None,
            "partial_compute",
            policy,
            self._onedal_params,
            self._partial_result,
            X_table,
            weights_table,
        )
