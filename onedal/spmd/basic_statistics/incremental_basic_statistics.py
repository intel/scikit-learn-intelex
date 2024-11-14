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

from daal4py.sklearn._utils import get_dtype

from ...basic_statistics import (
    IncrementalBasicStatistics as base_IncrementalBasicStatistics,
)
from ...common._backend import bind_default_backend, bind_spmd_backend
from ...datatypes import _convert_to_supported, to_table


class IncrementalBasicStatistics(base_IncrementalBasicStatistics):
    @bind_default_backend("basic_statistics", lookup_name="_get_policy")
    def _get_default_policy(self, queue, *data): ...

    @bind_spmd_backend("basic_statistics", lookup_name="_get_policy")
    def _get_spmd_policy(self, queue, *data): ...

    @bind_spmd_backend("basic_statistics")
    def compute(self, *args, **kwargs): ...

    @bind_spmd_backend("basic_statistics")
    def finalize_compute(self, *args, **kwargs): ...

    def partial_fit(self, *args, **kwargs):
        # base class partial_fit is using `compute()`, which requires host or parallel policy, but not SPMD
        self._get_policy = self._get_default_policy
        return super().partial_fit(*args, **kwargs)

    def finalize_fit(self, *args, **kwargs):
        # base class finalize_fit is using `finalize_compute()`, which requires SPMD policy
        self._get_policy = self._get_spmd_policy
        return super().finalize_fit(*args, **kwargs)
