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


class BaseLinearRegression(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, fit_intercept, copy_X, algorithm):
        self.fit_intercept = fit_intercept
        self.algorithm = algorithm
        self.copy_X = copy_X

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_onedal_params(self, dtype=np.float32):
        intercept = 'intercept|' if self.fit_intercept else ''
        return {
            'fptype': 'float' if dtype == np.float32 else 'double',
            'method': self.algorithm, 'intercept': self.fit_intercept,
            'result_option': (intercept + 'coefficients'),
        }

    def _fit(self, X, y, module, queue):
        policy = self._get_policy(queue, X, y)

        X_loc, y_loc = X, y
        if not isinstance(X, np.ndarray):
            X_loc = np.asarray(X)

        dtype = get_dtype(X_loc)
        if dtype not in [np.float32, np.float64]:
            dtype = np.float64
            X_loc = X_loc.astype(dtype, copy=self.copy_X)

        y_loc = np.asarray(y_loc).astype(dtype=dtype)

        # Finiteness is checked in the sklearnex wrapper
        X_loc, y_loc = _check_X_y(
            X_loc, y_loc, force_all_finite=False, accept_2d_y=True)

        self.n_features_in_ = _num_features(X_loc, fallback_1d=True)

        X_loc, y_loc = _convert_to_supported(policy, X_loc, y_loc)
        params = self._get_onedal_params(get_dtype(X_loc))
        X_table, y_table = to_table(X_loc, y_loc)

        result = module.train(policy, params, X_table, y_table)

        self._onedal_model = result.model

        coeff = from_table(result.model.packed_coefficients)
        self.coef_, self.intercept_ = coeff[:, 1:], coeff[:, 0]

        if self.coef_.shape[0] == 1 and y_loc.ndim == 1:
            self.coef_ = self.coef_.ravel()
            self.intercept_ = self.intercept_[0]

        return self

    def _create_model(self, module, policy):
        m = module.model()

        coefficients = self.coef_
        dtype = get_dtype(coefficients)
        if not isinstance(coefficients, np.ndarray):
            coefficients = np.asarray(coefficients, dtype=dtype)

        if coefficients.ndim == 2:
            n_features_in = coefficients.shape[1]
            n_targets_in = coefficients.shape[0]
        else:
            n_features_in = coefficients.size
            n_targets_in = 1

        intercept = self.intercept_
        if isinstance(intercept, Number):
            assert n_targets_in == 1
        else:
            if not isinstance(intercept, np.ndarray):
                intercept = np.asarray(intercept, dtype=dtype)
            assert n_targets_in == intercept.size

        intercept = _check_array(intercept, dtype=[np.float64, np.float32],
                                 force_all_finite=True, ensure_2d=False)
        coefficients = _check_array(
            coefficients,
            dtype=[
                np.float64,
                np.float32],
            force_all_finite=True, ensure_2d=False)

        coefficients, intercept = make2d(coefficients), make2d(intercept)
        coefficients = coefficients.T if n_targets_in == 1 else coefficients

        assert coefficients.shape == (n_targets_in, n_features_in)
        assert intercept.shape == (n_targets_in, 1)

        desired_shape = (n_targets_in, n_features_in + 1)
        packed_coefficients = np.zeros(desired_shape, dtype=dtype)

        packed_coefficients[:, 1:] = coefficients
        if self.fit_intercept:
            packed_coefficients[:, 0][:, np.newaxis] = intercept

        packed_coefficients = _convert_to_supported(policy, packed_coefficients)

        m.packed_coefficients = to_table(packed_coefficients)

        self._onedal_model = m

        return m

    def _predict(self, X, module, queue):
        _check_is_fitted(self)

        policy = self._get_policy(queue, X)

        if isinstance(X, np.ndarray):
            X_loc = np.asarray(X)
        else:
            X_loc = X

        # Finiteness is checked in the sklearnex wrapper
        X_loc = _check_array(X_loc, dtype=[np.float64, np.float32],
                             force_all_finite=False, ensure_2d=False)
        _check_n_features(self, X_loc, False)

        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(module, policy)

        X_loc = make2d(X_loc)
        X_loc = _convert_to_supported(policy, X_loc)
        params = self._get_onedal_params(get_dtype(X_loc))

        X_table = to_table(X_loc)
        result = module.infer(policy, params, model, X_table)
        y = from_table(result.responses)

        if not isinstance(self.coef_, np.ndarray):
            coefficients = np.asarray(self.coef_)
        else:
            coefficients = self.coef_

        if y.shape[1] == 1 and coefficients.ndim == 1:
            return y.ravel()
        else:
            return y


class LinearRegression(RegressorMixin, BaseLinearRegression):
    """
    Linear Regression oneDAL implementation.
    """

    def __init__(
            self,
            fit_intercept=True,
            copy_X=False,
            *,
            algorithm='norm_eq',
            **kwargs):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, algorithm=algorithm)

    def fit(self, X, y, queue=None):
        return super()._fit(X, y, _backend.linear_model.regression, queue)

    def predict(self, X, queue=None):
        y = super()._predict(X, _backend.linear_model.regression, queue)
        return y
