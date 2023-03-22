#===============================================================================
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
#===============================================================================

from abc import ABC

from onedal.spmd.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.spmd.ensemble import RandomForestRegressor as onedal_RandomForestRegressor

from ...preview.ensemble.forest import RandomForestClassifier as RandomForestClassifier_Batch
from ...preview.ensemble.forest import RandomForestRegressor as RandomForestRegressor_Batch


class BaseForestSPMD(ABC):
    def onedal_classifier(self, **onedal_params):
        return onedal_RandomForestClassifier(**onedal_params)

    def onedal_regressor(self, **onedal_params):
        return onedal_RandomForestRegressor(**onedal_params)


class RandomForestClassifier(BaseForestSPMD, RandomForestClassifier_Batch):

    # TODO:
    # update cpu/gpu support. Add error raise if not supported.
    def _onedal_cpu_supported(self, method_name, *data):
        return True

    def _onedal_gpu_supported(self, method_name, *data):
        return True


class RandomForestRegressor(BaseForestSPMD, RandomForestRegressor_Batch):

    # TODO:
    # update cpu/gpu support. Add error raise if not supported.
    def _onedal_cpu_supported(self, method_name, *data):
        return True

    def _onedal_gpu_supported(self, method_name, *data):
        return True
