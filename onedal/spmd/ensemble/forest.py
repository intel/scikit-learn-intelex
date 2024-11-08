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

import onedal._spmd_backend.decision_forest as onedal_backend
from onedal.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
from onedal.ensemble import RandomForestRegressor as RandomForestRegressor_Batch

from .._base import BaseEstimatorSPMD


class RandomForestClassifier(BaseEstimatorSPMD, RandomForestClassifier_Batch):
    _backend = onedal_backend


class RandomForestRegressor(BaseEstimatorSPMD, RandomForestRegressor_Batch):
    _backend = onedal_backend
