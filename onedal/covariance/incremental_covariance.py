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
from sklearn.utils import check_array

from daal4py.sklearn._utils import daal_check_version, get_dtype, make2d
from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes import _convert_to_supported, from_table, to_table


class IncrementalEmpiricalCovariance:
    def __init__(self, method="dense", bias=False):
        self.method = method
        self.bias = bias
        self._partial_result = _backend.covariance.partial_compute_result()

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_onedal_params(self, dtype=np.float32):
        params = {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": self.method,
        }
        if daal_check_version((2024, "P", 1)):
            params["bias"] = self.bias
        return params

    def partial_compute(self, X, queue=None):
        if not hasattr(self, "_policy"):
            self._policy = self._get_policy(queue, X)
        if not hasattr(self, "_dtype"):
            self._dtype = get_dtype(X)
        X = check_array(X, dtype=[np.float64, np.float32])
        X = make2d(X)
        types = [np.float32, np.float64]
        if get_dtype(X) not in types:
            X = X.astype(np.float64)
        X = _convert_to_supported(self._policy, X)
        params = self._get_onedal_params(self._dtype)
        module = _backend.covariance
        table_X = to_table(X)
        self._partial_result = module.partial_compute(
            self._policy, params, self._partial_result, table_X
        )

    def finalize_compute(self, queue=None):
        params = self._get_onedal_params(self._dtype)
        module = _backend.covariance
        result = module.finalize_compute(self._policy, params, self._partial_result)
        if daal_check_version((2024, "P", 1)) or (not self.bias):
            self.covariance_ = from_table(result.cov_matrix)
        else:
            n_rows = self._partial_result.partial_n_rows
            self.covariance_ = from_table(result.cov_matrix) * (n_rows - 1) / n_rows

        self.location_ = from_table(result.means).ravel()

        return self
