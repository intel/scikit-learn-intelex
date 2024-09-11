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

from onedal.neighbors import KNeighborsClassifier as KNeighborsClassifier_Batch
from onedal.neighbors import KNeighborsRegressor as KNeighborsRegressor_Batch

from ..._device_offload import support_usm_ndarray
from .._base import BaseEstimatorSPMD


class KNeighborsClassifier(BaseEstimatorSPMD, KNeighborsClassifier_Batch):
    @support_usm_ndarray()
    def fit(self, X, y, queue=None):
        return super().fit(X, y, queue=queue)

    @support_usm_ndarray()
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

    @support_usm_ndarray()
    def predict_proba(self, X, queue=None):
        raise NotImplementedError("predict_proba not supported in distributed mode.")

    @support_usm_ndarray()
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)


class KNeighborsRegressor(BaseEstimatorSPMD, KNeighborsRegressor_Batch):
    @support_usm_ndarray()
    def fit(self, X, y, queue=None):
        if queue is not None and queue.sycl_device.is_gpu:
            return super()._fit(X, y, queue=queue)
        else:
            raise ValueError(
                "SPMD version of kNN is not implemented for "
                "CPU. Consider running on it on GPU."
            )

    @support_usm_ndarray()
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)

    @support_usm_ndarray()
    def predict(self, X, queue=None):
        return self._predict_gpu(X, queue=queue)

    def _get_onedal_params(self, X, y=None):
        params = super()._get_onedal_params(X, y)
        if "responses" not in params["result_option"]:
            params["result_option"] += "|responses"
        return params


class NearestNeighbors(BaseEstimatorSPMD):
    @support_usm_ndarray()
    def fit(self, X, y, queue=None):
        return super().fit(X, y, queue=queue)

    @support_usm_ndarray()
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)
