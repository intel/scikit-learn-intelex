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

from onedal.spmd.cluster import DBSCAN as onedal_DBSCAN

from ...cluster import DBSCAN as DBSCAN_Batch


class BaseDBSCANspmd(ABC):
    def _onedal_dbscan(self, **onedal_params):
        return onedal_DBSCAN(**onedal_params)


class DBSCAN(BaseDBSCANspmd, DBSCAN_Batch):
    __doc__ = DBSCAN_Batch.__doc__

    def _onedal_cpu_supported(self, method_name, *data):
        # TODO:
        # check which methods supported SPMD interface on CPU.
        ready = super()._onedal_cpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

    def _onedal_gpu_supported(self, method_name, *data):
        ready = super()._onedal_gpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready
