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

from onedal import _spmd_backend
from onedal.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
from onedal.ensemble import RandomForestRegressor as RandomForestRegressor_Batch

from .._common import BaseEstimatorSPMD


class RandomForestClassifier(BaseEstimatorSPMD, RandomForestClassifier_Batch):
    pass


class RandomForestRegressor(BaseEstimatorSPMD, RandomForestRegressor_Batch):
    def fit(self, X, y, sample_weight=None, queue=None):
        if sample_weight is not None:
            if hasattr(sample_weight, "__array__"):
                sample_weight[sample_weight == 0.0] = 1.0
            sample_weight = [sample_weight]
        return super()._fit(
            X,
            y,
            sample_weight,
            _spmd_backend.decision_forest.regression,
            queue,
        )

    def predict(self, X, queue=None):
        return (
            super()._predict(X, _spmd_backend.decision_forest.regression, queue).ravel()
        )
