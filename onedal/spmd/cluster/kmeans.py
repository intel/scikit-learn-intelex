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

from abc import ABC

from onedal.cluster import KMeans as KMeans_Batch

from ..._device_offload import support_usm_ndarray
from ...common._spmd_policy import _get_spmd_policy


class BaseKMeansSPMD(ABC):
    def _get_policy(self, queue, *data):
        return _get_spmd_policy(queue)


class KMeans(BaseKMeansSPMD, KMeans_Batch):
    @support_usm_ndarray()
    def fit(self, X, queue=None):
        return super().fit(X, queue)

    @support_usm_ndarray()
    def predict(self, X, queue=None):
        return super().predict(X, queue)

    @support_usm_ndarray()
    def fit_predict(self, X, queue=None):
        return super().fit_predict(X, queue)

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, queue=None):
        return super().fit_transform(X, queue)
