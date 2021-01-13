#===============================================================================
# Copyright 2014-2021 Intel Corporation
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
#===============================================================================

import numpy as np
from scipy import sparse as sp
from sklearn.utils import check_array, check_X_y
from sklearn.base import RegressorMixin
from sklearn.linear_model.ridge import _BaseRidge
from sklearn.linear_model.ridge import Ridge as Ridge_original

import daal4py
from .._utils import (make2d, getFPType, get_patch_message)
import logging


def _daal4py_fit(self, X, y_):
    X = make2d(X)
    y = make2d(y_)

    _fptype = getFPType(X)

    ridge_params = np.asarray(self.alpha, dtype=X.dtype)
    if ridge_params.size != 1 and ridge_params.size != y.shape[1]:
        raise ValueError("alpha length is wrong")
    ridge_params = ridge_params.reshape((1,-1))

    ridge_alg = daal4py.ridge_regression_training(
        fptype=_fptype,
        method='defaultDense',
        interceptFlag=(self.fit_intercept is True),
        ridgeParameters=ridge_params
    )
    ridge_res = ridge_alg.compute(X, y)

    ridge_model = ridge_res.model
    self.daal_model_ = ridge_model
    coefs = ridge_model.Beta

    self.intercept_ = coefs[:,0].copy(order='C')
    self.coef_ = coefs[:,1:].copy(order='C')

    if self.coef_.shape[0] == 1 and y_.ndim == 1:
        self.coef_ = np.ravel(self.coef_)
        self.intercept_ = self.intercept_[0]

    return self

def _daal4py_predict(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)

    ridge_palg = daal4py.ridge_regression_prediction(
        fptype=_fptype,
        method='defaultDense'
    )
    ridge_res = ridge_palg.compute(X, self.daal_model_)

    res = ridge_res.prediction

    if res.shape[1] == 1 and self.coef_.ndim == 1:
        res = np.ravel(res)
    return res


def _fit_ridge(self, X, y, sample_weight=None):
    """Fit Ridge regression model

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training data

    y : array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values

    sample_weight : float or numpy array of shape [n_samples]
        Individual weights for each sample

    Returns
    -------
    self : returns an instance of self.
    """
    X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=[np.float64, np.float32],
            multi_output=True, y_numeric=True)
    self.sample_weight_ = sample_weight
    self.fit_shape_good_for_daal_ = True if X.shape[0] >= X.shape[1] else False
    if (not self.solver == 'auto' or
            sp.issparse(X) or
            not self.fit_shape_good_for_daal_ or
            not (X.dtype == np.float64 or X.dtype == np.float32) or
            sample_weight is not None):
        if hasattr(self, 'daal_model_'):
            del self.daal_model_
        logging.info("sklearn.linear_model.Ridge.fit: " + get_patch_message("sklearn"))
        return super(Ridge, self).fit(X, y, sample_weight=sample_weight)
    logging.info("sklearn.linear_model.Ridge.fit: " + get_patch_message("daal"))
    self.n_iter_ = None
    return _daal4py_fit(self, X, y)

def _predict_ridge(self, X):
    """Predict using the linear model

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = (n_samples, n_features)
        Samples.

    Returns
    -------
    C : array, shape = (n_samples,)
        Returns predicted values.
    """
    X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
    good_shape_for_daal = True if X.ndim <= 1 else True if X.shape[0] >= X.shape[1] else False

    if (not self.solver == 'auto' or
            not hasattr(self, 'daal_model_') or
            sp.issparse(X) or
            not good_shape_for_daal or
            not (X.dtype == np.float64 or X.dtype == np.float32) or
            (hasattr(self, 'sample_weight_') and self.sample_weight_ is not None)):
        logging.info("sklearn.linear_model.Ridge.predict: " + get_patch_message("sklearn"))
        return self._decision_function(X)
    logging.info("sklearn.linear_model.Ridge.predict: " + get_patch_message("daal"))
    return _daal4py_predict(self, X)

class Ridge(Ridge_original, _BaseRidge):
    __doc__ = Ridge_original.__doc__

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        return _fit_ridge(self, X, y, sample_weight=sample_weight)

    def predict(self, X):
        return _predict_ridge(self, X)
