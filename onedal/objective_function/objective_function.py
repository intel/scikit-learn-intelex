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

from ..common._policy import _get_policy

from ..datatypes._data_conversion import (
    from_table,
    to_table,
    _convert_to_supported,
    _convert_to_dataframe)
from onedal import _backend


class BaseLogisticLoss(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, algorithm, queue, module):
        self.algorithm = self.algorithm
        self.queue = queue
        self.module = module
    
    @staticmethod
    def get_all_result_options():
        return ["value", "gradient", "hessian"]

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_result_options(self, options):
        if options == "all":
            options = self.get_all_result_options()
        if isinstance(options, list):
            options = "|".join(options)
        assert isinstance(options, str)
        return options

    def _get_onedal_params(self, options, L1 = 0.0, L2 = 0.0, intercept = True, dtype=np.float32):
        options = self._get_result_options(options)
        return {
            'fptype': 'float' if dtype == np.float32 else 'double',
            'method': self.algorithm, 'result_option': options,
            'l1_coef' : L1, 'l2_coef' : L2, 'intercept' : intercept
        }


    def _compute(self, X, y, coef, options, l2_reg_strength=0.0, fit_intercept=True):
        
        policy = self._get_policy(self.queue, X, y, coef)
        ftype = X.dtype
        params = self._get_onedal_params(options, L2 = l2_reg_strength, intercept = fit_intercept, dtype = self.ftype)

        X_loc, y_loc, coef_loc = _convert_to_dataframe(self.policy, X, y, coef)
        X_loc, y_loc, coef_loc = _convert_to_supported(self.policy, X_loc, y_loc, coef_loc)

        X_table, y_table, coef_table = to_table(X_loc, y_loc, coef_loc)

        
        result = self.module.train(policy, params, X_table, coef_table, y_table)
        
        options = self._get_result_options(options)
        options = options.split("|")

        res = {opt: getattr(result, opt) for opt in options}

        return {k: from_table(v).ravel() for k, v in res.items()}


    



class LogisticLoss(BaseLogisticLoss):
    def __init__(
            self,
            *,
            algorithm="by_default",
            queue = None,
            **kwargs):
        super().__init__(algorithm, queue, _backend.objective_function)

    def loss(self, coef, X, y, l2_reg_strength=0.0, fit_intercept=True):
        return super()._compute(X, y, coef, "value", l2_reg_strength, fit_intercept)["value"]
    
    def loss_gradient(self, coef, X, y, l2_reg_strength=0.0, fit_intercept=True):
        res = super()._compute(X, y, coef, ["value", "gradient"], l2_reg_strength, fit_intercept)
        return res["value"], res["gradient"]

    def gradient(self, coef, X, y, l2_reg_strength=0.0, fit_intercept=True):
        return super()._compute(X, y, coef, "gradient", l2_reg_strength, fit_intercept)["gradient"]

    def gradient_hessian(self, coef, X, y, l2_reg_strength=0.0, fit_intercept=True):
        res = super()._compute(X, y, coef, ["gradient", "hessian"], l2_reg_strength, fit_intercept)
        return res["gradient"], res["hessian"]

    def gradient_hessian_product(self, coef, X, y, l2_reg_strength=0.0, fit_intercept=True):
        res = super()._compute(X, y, coef, ["gradient", "hessian"], l2_reg_strength, fit_intercept)
        
        H = res["hessian"]

        def hessp(s):
            return H @ s
        
        return res["gradient"], hessp