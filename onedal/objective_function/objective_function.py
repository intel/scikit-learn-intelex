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


class BaseObjectiveFunction(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, algorithm, queue, objective_function_method):
        self.algorithm = algorithm
        self.queue = queue
        self.func = objective_function_method

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

    def _get_onedal_params(self, options, L1=0.0, L2=0.0, intercept=True, dtype=np.float32):
        options = self._get_result_options(options)
        return {
            'fptype': 'float' if dtype == np.float32 else 'double',
            'method': self.algorithm, 'result_option': options,
            'l1_coef': L1, 'l2_coef': L2, 'intercept': intercept
        }

    def _compute(self, X, y, coef, options, l2_reg_strength=0.0, fit_intercept=True):

        policy = self._get_policy(self.queue, X, y, coef)
        ftype = X.dtype
        params = self._get_onedal_params(
            options, L2=l2_reg_strength, intercept=fit_intercept, dtype=ftype)

        y = y.astype(np.int32)

        # in python interface intercept-coef is at the end of the array
        # in onedal interface coef array always has the size p + 1
        # coef[0] is considered as intercept-coef or ignored if fit_intercept=False

        coef = coef.reshape(-1)
        if (fit_intercept):
            coef = np.hstack([coef[-1], coef[:-1]])
        else:
            coef = np.hstack([0.0, coef])
        coef = coef.astype(X.dtype)

        X_loc, y_loc, coef_loc = _convert_to_dataframe(policy, X, y, coef)
        X_loc, y_loc, coef_loc = _convert_to_supported(
            policy, X_loc, y_loc, coef_loc)

        X_table, coef_table = to_table(X_loc, coef_loc)
        y_table = to_table(y_loc)

        result = self.func(policy, params, X_table, coef_table, y_table)

        options = self._get_result_options(options)
        options = options.split("|")

        res = {opt: getattr(result, opt) for opt in options}

        return {k: from_table(v).ravel() for k, v in res.items()}


class LogisticLoss(BaseObjectiveFunction):
    def __init__(
            self,
            *,
            algorithm="by_default",
            queue=None,
            l2_reg_strength=0.0, 
            fit_intercept=True,
            **kwargs):
        self.fit_intercept = fit_intercept
        self.l2_reg_strength = l2_reg_strength
        super().__init__(algorithm, queue, _backend.objective_function.compute.logloss)

    def __fix_gradient(self, grad, fit_intercept):
        if (fit_intercept):
            return np.hstack([grad[1:], grad[0]])
        else:
            return grad[1:]

    def __fix_hessian(self, hess, num_params, fit_intercept):
        hess = hess.reshape(num_params + 1, num_params + 1)
        if (fit_intercept):
            return np.hstack((np.vstack([hess[1:,1:], hess[0,1:]]), np.hstack([hess[0,1:], hess[0][0]]).reshape(-1, 1)))
        else:
            return hess[1:,1:]

    def loss(self, coef, X, y):
        return super()._compute(X, y, coef, "value", self.l2_reg_strength, self.fit_intercept)["value"]

    def loss_gradient(self, coef, X, y):
        res = super()._compute(
            X, y, coef, ["value", "gradient"], self.l2_reg_strength, self.fit_intercept)
        return (res["value"], self.__fix_gradient(res["gradient"], self.fit_intercept))

    def gradient(self, coef, X, y):
        grad = super()._compute(X, y, coef, "gradient", self.l2_reg_strength, self.fit_intercept)["gradient"]
        return self.__fix_gradient(grad, self.fit_intercept)

    def gradient_hessian(self, coef, X, y):
        res = super()._compute(
            X, y, coef, ["gradient", "hessian"], self.l2_reg_strength, self.fit_intercept)
        grad = self.__fix_gradient(res["gradient"], self.fit_intercept)
        hess = self.__fix_hessian(res["hessian"], X.shape[1], self.fit_intercept)
        flag = (res["hessian"] <= 0.0).sum() * 2 >= res["hessian"].shape[0]
        return (grad, hess, flag)
    

