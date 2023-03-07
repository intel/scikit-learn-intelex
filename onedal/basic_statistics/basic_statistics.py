# ===============================================================================
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
# ===============================================================================

from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod

import numpy as np
from numbers import Number

from daal4py.sklearn._utils import (get_dtype, make2d)
from ..datatypes import (
    _check_X_y,
    _num_features,
    _check_array,
    _get_2d_shape,
    _check_n_features,
    _convert_to_supported)

from ..common._mixin import RegressorMixin
from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted
from ..datatypes._data_conversion import from_table, to_table
from onedal import _backend




class BaseBasicStatistics(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, result_options, algorithm):
        self.fit_intercept = fit_intercept
        self.algorithm = algorithm
        self.copy_X = copy_X

    @staticmethod
    def get_all_result_options():
        return [ "min", "max", "sum", "mean",
                 "variance", "variation", "sum_squares",
                 "standard_deviation", "sum_squares_centered",
                 "second_order_raw_moment"] 

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_result_options(self, options):
        if options == "all":
            options = self.get_all_result_options()
        if isinstance(options, list):
            options = "|".join(options)
        assert isinstance(options, str)
        return options

    def _get_onedal_params(self, options, dtype=np.float32):
        return {
            'fptype': 'float' if dtype is np.float32 else 'double',
            'method': self.algorithm, 'result_option': self.options,
        }

    @staticmethod
    def _convert_to_dataframe(x):
        if x is None:
            return None

        is_numpy = isinstance(x, np.ndarray)
        if not is_numpy:
            return np.asarray(x)
        else:
            return x

    def _compute(self, data, weights, queue):
        policy = self._get_policy(queue, data, weights)

        data_loc, weights_loc = data, weights
        is_numpy_data = isinstance(data, np.ndarray)

        if not is_numpy_data:
            data_loc = np.asarray(data)


        

        dtype = X_loc.dtype
        if dtype not in [np.float32, np.float64]:
            X_loc = X_loc.astype(np.float64, copy=self.copy_X)
            

        y_loc = np.asarray(y_loc).astype(dtype=dtype)

        params = self._get_onedal_params(dtype)

        self.n_features_in_ = _num_features(X_loc, fallback_1d=True)

        X_loc, y_loc = _convert_to_supported(policy, X_loc, y_loc)
        X_table, y_table = to_table(X_loc, y_loc)

        result = module.train(policy, params, X_table, weights)

        return self


class BasicStatistics(BaseBasicStatistics):
    """
    Linear Regression oneDAL implementation.
    """

    def __init__(
            self,
            result_options="all",
            *,
            algorithm="by_default'
        "",
            **kwargs):
        super().__init__(result_options, algorithm)

    def compute(self, data, weights=None, queue=None):
        return super()._compute(data, weights, queue)
