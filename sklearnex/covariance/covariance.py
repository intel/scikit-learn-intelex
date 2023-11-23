#!/usr/bin/env python
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

from onedal.covariance import EmpiricalCovariance as onedal_EmpiricalCovariance
from sklearn.covariance import EmpiricalCovariance as sklearn_EmpiricalCovariance

from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain
from daal4py.sklearn._utils import sklearn_check_version
from scipy import sparse as sp

class EmpiricalCovariance(sklearn_EmpiricalCovariance):
    def __init__(self, *, store_precision=True, assume_centered=False):
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")
        self.covariance_ = self._onedal_estimator.covariance_
        self.location_ = self._onedal_estimator.location_

    def _onedal_covariance(self, **onedal_params):
        return onedal_EmpiricalCovariance(**onedal_params)

    def _onedal_fit(self, X, queue=None):
        onedal_params = {
            "method": "dense",
            "bias": True,
        }

        self._onedal_estimator = self._onedal_covariance(**onedal_params)
        self._onedal_estimator.fit(X, queue=queue)
        self._save_attributes()

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.covariance.{class_name}.{method_name}"
        )
        if method_name == "fit":
            X, = data
            patching_status.and_conditions(
                [
                    (
                        self.assume_centered == False,
                        "assume_centered parameter is not supported on oneDAL side"
                    ),
                    (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
                ]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    
    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def fit(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": sklearn_EmpiricalCovariance.fit,
            },
            X,
        )

        return self