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

import onedal._spmd_backend.logistic_regression.classification as onedal_backend
from onedal.linear_model import LogisticRegression as LogisticRegression_Batch

from ..._device_offload import support_input_format
from .._base import BaseEstimatorSPMD


class LogisticRegression(BaseEstimatorSPMD, LogisticRegression_Batch):
    _backend = onedal_backend

    @support_input_format()
    def fit(self, X, y, queue=None):
        return super().fit(X, y, queue=queue)

    @support_input_format()
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

    @support_input_format()
    def predict_proba(self, X, queue=None):
        return super().predict_proba(X, queue=queue)

    @support_input_format()
    def predict_log_proba(self, X, queue=None):
        return super().predict_log_proba(X, queue=queue)
