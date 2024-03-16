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

from onedal.cluster import KMeans as KMeans_Batch
from onedal.cluster import KMeansInit as KMeansInit_Batch
from onedal.spmd.basic_statistics import BasicStatistics

from ..._device_offload import support_usm_ndarray
from .._base import BaseEstimatorSPMD


class KMeansInit(BaseEstimatorSPMD, KMeansInit_Batch):
    """
    KMeansInit oneDAL implementation for SPMD iface.
    """

    pass


class KMeans(BaseEstimatorSPMD, KMeans_Batch):
    def _get_basic_statistics_backend(self, result_options):
        return BasicStatistics(result_options)

    def _get_kmeans_init(self, cluster_count, seed, algorithm):
        return KMeansInit(cluster_count=cluster_count, seed=seed, algorithm=algorithm)

    @support_usm_ndarray()
    def fit(self, X, y=None, queue=None):
        return super().fit(X, queue=queue)

    @support_usm_ndarray()
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

    @support_usm_ndarray()
    def fit_predict(self, X, y=None, queue=None):
        return super().fit_predict(X, queue=queue)

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, queue=None):
        return super().fit_transform(X, queue=queue)
