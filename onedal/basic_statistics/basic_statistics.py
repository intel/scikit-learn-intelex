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

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np

from ..common._base import BaseEstimator
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _is_csr


class BasicStatistics(BaseEstimator, metaclass=ABCMeta):
    """
    Basic Statistics oneDAL implementation.
    """

    def __init__(self, result_options="all", algorithm="by_default"):
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

    @property
    def options(self):
        if self._options == "all":
            return self.get_all_result_options()
        return self._options

    @options.setter
    def options(self, options):
        # options always to be an iterable
        self._options = options.split("|") if isinstance(options, str) else options

    def _get_onedal_params(self, is_csr, dtype=np.float32):
        return {
            "fptype": dtype,
            "method": "sparse" if is_csr else self.algorithm,
            "result_option": "|".join(self.options),
        }

    def fit(self, data, sample_weight=None, queue=None):
        policy = self._get_policy(queue, data, sample_weight)

        is_csr = _is_csr(data)

        data, sample_weight = _convert_to_supported(policy, data, sample_weight)
        is_single_dim = data.ndim == 1
        data_table, weights_table = to_table(data, sample_weight)

        dtype = data_table.dtype
        module = self._get_backend("basic_statistics")
        params = self._get_onedal_params(is_csr, data_table.dtype)
        result = module.compute(policy, params, data_table, weights_table)

        for opt in self.options:
            value = from_table(getattr(result, opt))[:, 0]  # two-dimensional table [n, 1]
            if is_single_dim:
                setattr(self, opt, value[0])
            else:
                setattr(self, opt, value)

        return self
