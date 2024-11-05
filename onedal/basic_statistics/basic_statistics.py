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

import numpy as np

from .._config import _get_config
from ..common._base import BaseEstimator
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import _is_csr
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import _check_array


class BaseBasicStatistics(BaseEstimator, metaclass=ABCMeta):
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

    def _get_onedal_params(self, is_csr, dtype=np.float32):
        options = self._get_result_options(self.options)
        return {
            "fptype": "float" if dtype == np.float32 else "double",
            "method": "sparse" if is_csr else self.algorithm,
            "result_option": options,
        }


class BasicStatistics(BaseBasicStatistics):
    """
    Basic Statistics oneDAL implementation.
    """

    def __init__(self, result_options="all", algorithm="by_default"):
        super().__init__(result_options, algorithm)

    def fit(self, data, sample_weight=None, queue=None):
        is_csr = _is_csr(data)

        use_raw_input = _get_config().get("use_raw_input", False) is True

        # All data should use the same sycl queue
        if use_raw_input and _get_sycl_namespace(data)[0] is not None:
            queue = data.sycl_queue

        if not use_raw_input:
            if data is not None and not is_csr:
                data = _check_array(data, ensure_2d=False)
            if sample_weight is not None:
                sample_weight = _check_array(sample_weight, ensure_2d=False)

        # TODO
        # use xp for dtype.
        policy = self._get_policy(queue, data, sample_weight)
        data, sample_weight = _convert_to_supported(policy, data, sample_weight)

        data_table = to_table(data, sua_iface=_get_sycl_namespace(data)[0])
        weights_table = to_table(sample_weight, sua_iface=_get_sycl_namespace(sample_weight)[0])

        dtype = data.dtype
        raw_result = self._compute_raw(data_table, weights_table, policy, dtype, is_csr)
        for opt, raw_value in raw_result.items():
            value = from_table(raw_value).ravel()
            setattr(self, opt, value[0]) if data.ndim == 1 else setattr(self, opt, value)

        return self

    def _compute_raw(
        self, data_table, weights_table, policy, dtype=np.float32, is_csr=False
    ):
        module = self._get_backend("basic_statistics")
        params = self._get_onedal_params(is_csr, dtype)
        result = module.compute(policy, params, data_table, weights_table)
        options = self._get_result_options(self.options).split("|")

        return {opt: getattr(result, opt) for opt in options}
