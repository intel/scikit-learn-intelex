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

from onedal.decomposition.pca import PCA as PCABatch

from ..._device_offload import support_input_format
from .._base import BaseEstimatorSPMD
import onedal._spmd_backend.decomposition.dim_reduction as onedal_backend


class PCA(BaseEstimatorSPMD, PCABatch):
    _backend = onedal_backend

    @support_input_format()
    def fit(self, X, y=None, queue=None):
        return super().fit(X, queue=queue)
