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

from abc import ABC

from onedal import _spmd_backend

from ..common._base import _get_backend
from ..common._spmd_policy import _get_spmd_policy


class BaseEstimatorSPMD(ABC):
    def _get_backend(self, module, submodule=None, method=None, *args, **kwargs):
        return _get_backend(_spmd_backend, module, submodule, method, *args, **kwargs)

    def _get_policy(self, queue, *data):
        return _get_spmd_policy(queue)
