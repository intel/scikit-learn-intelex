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

from daal4py.sklearn._utils import sklearn_check_version
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
from enum import Enum
from numbers import Number

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _check_X_y,
    _check_array,
    _check_n_features
)

import inspect

from ..common._mixin import ClassifierMixin, RegressorMixin
from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted
from ..datatypes._data_conversion import from_table, to_table
from onedal import _backend


class BaseLinearRegression(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, fit_intercept, algorithm):
        self.fit_intercept = fit_intercept
        self.algorithm = algorithm
        self._sparse = False

    def _get_onedal_params(self, dtype=np.float32):
        intercept = 'intercept|' if self.fit_intercept else ''
        return {
            'fptype': 'float' if dtype is np.float32 else 'double',
            'method': self.algorithm, 'intercept': self.fit_intercept,
            'result_option': (intercept + 'coefficients'),
        }

    def _fit(self, X, y, module, queue):
        X = _check_array(
            X,
            dtype=[
                np.float64,
                np.float32],
            ensure_2d=False,
            force_all_finite=True,
            accept_sparse='csr')
        y = _check_array(
            y,
            dtype=[
                np.float64,
                np.float32],
            ensure_2d=False,
            force_all_finite=True,
            accept_sparse='csr')

        self._sparse = sp.isspmatrix(X)

        dtype = X.dtype
        policy = _get_policy(queue, X, y)
        params = self._get_onedal_params(dtype)

        result = module.train(policy, params,
                              to_table(X), to_table(y))

        self.coef_ = from_table(result.coefficients)
        self.n_targets_, self.n_features_in_ = self.coef_.shape
        if len(self.coef_.shape) == 2 and self.coef_.shape[0] == 1:
            assert self.coef_.shape[0] == 1
            self.coef_ = self.coef_.ravel()

        if self.fit_intercept:
            self.intercept_ = from_table(result.intercept)
        else:
            self.intercept_ = np.zeros(self.n_targets_)
        self.intercept_ = self.intercept_.ravel()

        if self._sparse:
            self.coef_ = sp.csr_matrix(self.coef_)

        self._onedal_model = result.model
        return self

    def _create_model(self, module):
        m = module.model()

        dtype = self.coef_.dtype
        is_multi_output = len(self.coef_.shape) == 2
        r_count = self.coef_.shape[0] if is_multi_output else 1
        intercept = self.intercept_ if self.fit_intercept \
            else np.zeros(r_count, dtype=dtype)
        if self.fit_intercept:
            assert intercept.shape == (r_count,)
        intercept = intercept.reshape(r_count, 1)

        coefficients = np.array(self.coef_)
        packed_coefficients = np.hstack((intercept, coefficients))
        m.packed_coefficients = to_table(packed_coefficients)

        return m

    def _predict(self, X, module, queue):
        _check_is_fitted(self)

        X = _check_array(X, dtype=[np.float64, np.float32],
                         force_all_finite=True, accept_sparse='csr')
        _check_n_features(self, X, False)

        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)

        if sp.issparse(X) and not self._sparse:
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)

        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)

        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(module)
        result = module.infer(policy, params, model, to_table(X))
        y = from_table(result.responses)

        if len(y.shape) == 2 and y.shape[1] == 1:
            return y.ravel()
        else:
            return y


class LinearRegression(RegressorMixin, BaseLinearRegression):
    """
    Epsilon--Support Vector Regression.
    """

    def __init__(self, fit_intercept=True, *, algorithm='norm_eq', **kwargs):
        super().__init__(fit_intercept=fit_intercept, algorithm=algorithm)

    def fit(self, X, y, queue=None):
        return super()._fit(X, y, _backend.linear_model.regression, queue)

    def predict(self, X, queue=None):
        y = super()._predict(X, _backend.linear_model.regression, queue)
        return y
