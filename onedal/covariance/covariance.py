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

from daal4py.sklearn._utils import get_dtype, make2d
from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes import _convert_to_supported, from_table, to_table


class BaseCovariance:
    def __init__(self):
        pass

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)
    
    def _get_onedal_params(self, dtype=np.float32):
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "dense",
        }

    def _fit(self, X, module, queue):
        policy = self._get_policy(queue, X)
        X = check_array(X, dtype=[np.float64, np.float32])
        X = make2d(X)
        types = [np.float32, np.float64]
        if get_dtype(X) not in types:
            X = X.astype(np.float64)
        X = _convert_to_supported(policy, X)
        dtype = get_dtype(X)
        params = self._get_onedal_params(dtype)
        result = module.compute(policy, params, to_table(X))
        self.covariance_ = from_table(result.cov_matrix)
        self.location_ = from_table(result.means).ravel()

        return self

class Covariance(BaseCovariance):
    def __init__(self):
        pass

    def fit(self, X, queue=None):
        return super()._fit(X, _backend.covariance, queue)
