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


from onedal.spmd.linear_model import (
    IncrementalLinearRegression as onedalSPMD_IncrementalLinearRegression,
)

from ...linear_model import (
    IncrementalLinearRegression as base_IncrementalLinearRegression,
)


class IncrementalLinearRegression(base_IncrementalLinearRegression):
    """
    Distributed incremental estimator for linear regression.
    Allows for distributed training of linear regression if data is split into batches.

    API is the same as for `sklearnex.linear_model.IncrementalLinearRegression`.
    """

    _onedal_incremental_linear = staticmethod(onedalSPMD_IncrementalLinearRegression)
