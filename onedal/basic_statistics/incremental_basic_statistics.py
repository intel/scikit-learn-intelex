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

import numpy as np

from daal4py.sklearn._utils import get_dtype
from onedal import _backend

from ..datatypes import _convert_to_supported, from_table, to_table
from .basic_statistics import BaseBasicStatistics


class IncrementalBasicStatistics(BaseBasicStatistics):
    def __init__(self, result_options="all", *, algorithm="by_default", **kwargs):
        super().__init__(result_options, algorithm)
        self._partial_result = self._module.partial_compute_result()

    def partial_fit(self, data, weights=None, queue=None):
        if not hasattr(self, "_policy"):
            self._policy = self._get_policy(queue, data)
        if not hasattr(self, "_onedal_params"):
            dtype = get_dtype(data)
            self._onedal_params = self._get_onedal_params(dtype)

        data, weights = _convert_to_supported(self._policy, data, weights)
        data_table, weights_table = to_table(data, weights)
        self._partial_result = self._module.partial_compute(
            self._policy,
            self._onedal_params,
            self._partial_result,
            data_table,
            weights_table,
        )

    def finalize_fit(self, queue=None):
        result = self._module.finalize_compute(
            self._policy, self._onedal_params, self._partial_result
        )
        options = self._get_result_options(self.options).split("|")
        for opt in options:
            setattr(self, opt, from_table(getattr(result, opt)).ravel())

        return self
