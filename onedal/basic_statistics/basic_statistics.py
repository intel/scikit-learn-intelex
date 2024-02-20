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

from abc import ABCMeta, abstractmethod
from numbers import Number

import numpy as np

from onedal import _backend

from ..common._base import BaseEstimator
from ..datatypes import _convert_to_supported, from_table, to_table


class BaseBasicStatistics(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, result_options, algorithm):
        self.options = result_options
        self.algorithm = algorithm

    @staticmethod
    def get_all_result_options():
        return [
            "min",
            "max",
            "sum",
            "mean",
            "variance",
            "variation",
            "sum_squares",
            "standard_deviation",
            "sum_squares_centered",
            "second_order_raw_moment",
        ]

    def _get_result_options(self, options):
        if options == "all":
            options = self.get_all_result_options()
        if isinstance(options, list):
            options = "|".join(options)
        assert isinstance(options, str)
        return options

    def _get_onedal_params(self, dtype=np.float32):
        options = self._get_result_options(self.options)
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": self.algorithm,
            "result_option": options,
        }

    def _compute_raw(self, data_table, weights_table, module, policy, dtype=np.float32):
        params = self._get_onedal_params(dtype)

        result = module.train(policy, params, data_table, weights_table)

        options = self._get_result_options(self.options)
        options = options.split("|")

        return {opt: getattr(result, opt) for opt in options}

    def _compute(self, data, weights, module, queue):
        policy = self._get_policy(queue, data, weights)

        if not (data is None):
            data = np.asarray(data)
        if not (weights is None):
            weights = np.asarray(weights)

        data, weights = _convert_to_supported(policy, data, weights)

        data_table, weights_table = to_table(data, weights)

        dtype = data.dtype
        res = self._compute_raw(data_table, weights_table, module, policy, dtype)

        return {k: from_table(v).ravel() for k, v in res.items()}


class BasicStatistics(BaseEstimator, BaseBasicStatistics):
    """
    Basic Statistics oneDAL implementation.
    """

    def __init__(self, result_options="all", *, algorithm="by_default", **kwargs):
        super().__init__(result_options, algorithm)

    def compute(self, data, weights=None, queue=None):
        return super()._compute(
            data, weights, self._get_backend("basic_statistics", "compute", None), queue
        )

    def compute_raw(self, data_table, weights_table, policy, dtype=np.float32):
        return super()._compute_raw(
            data_table,
            weights_table,
            self._get_backend("basic_statistics", "compute", None),
            policy,
            dtype,
        )
