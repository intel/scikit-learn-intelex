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
from scipy import linalg

from sklearn.linear_model._base import _rescale_data
from ..utils.validation import _daal_check_array, _daal_check_X_y
from ..utils.base import _daal_validate_data
from .._utils import sklearn_check_version
from .._device_offload import support_usm_ndarray
from sklearn.utils.fixes import sparse_lsqr
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import check_array

from sklearn.linear_model import LinearRegression as LinearRegression_original
if sklearn_check_version('1.0'):
    from sklearn.linear_model._base import _deprecate_normalize

try:
    from sklearn.utils._joblib import Parallel, delayed
except ImportError:
    from sklearn.externals.joblib import Parallel, delayed

import daal4py
from .._utils import (
    make2d,
    getFPType,
    get_patch_message,
    is_DataFrame,
    get_dtype)
import logging


def _daal4py_fit(self, X, y_):
    y = make2d(y_)
    X_fptype = getFPType(X)

    try:
        lr_algorithm = daal4py.linear_regression_training(
            fptype=X_fptype,
            interceptFlag=bool(self.fit_intercept),
            method='defaultDense'
        )
        lr_res = lr_algorithm.compute(X, y)
    except RuntimeError:
        # Normal system is not invertible, try QR
        try:
            lr_algorithm = daal4py.linear_regression_training(
                fptype=X_fptype,
                interceptFlag=bool(self.fit_intercept),
                method='qrDense'
            )
            lr_res = lr_algorithm.compute(X, y)
        except RuntimeError:
            # fall back on sklearn
            return None

    lr_model = lr_res.model
    self.daal_model_ = lr_model
    coefs = lr_model.Beta

    self.intercept_ = coefs[:, 0].copy(order='C')
    self.coef_ = coefs[:, 1:].copy(order='C')
    self.n_features_in_ = X.shape[1]
    self.rank_ = X.shape[1]
    self.singular_ = np.full((X.shape[1],), np.nan)

    if self.coef_.shape[0] == 1 and y_.ndim == 1:
        self.coef_ = np.ravel(self.coef_)
        self.intercept_ = self.intercept_[0]

    return self


def _daal4py_predict(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)
    lr_pred = daal4py.linear_regression_prediction(
        fptype=_fptype,
        method='defaultDense'
    )
    if sklearn_check_version('0.23'):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {X.shape[1]} features, '
                f'but LinearRegression is expecting '
                f'{self.n_features_in_} features as input')
    try:
        lr_res = lr_pred.compute(X, self.daal_model_)
    except RuntimeError:
        raise ValueError(
            f'Input data shape {X.shape} is inconsistent with the trained model'
        )
    res = lr_res.prediction
    if res.shape[1] == 1 and self.coef_.ndim == 1:
        res = np.ravel(res)

    return res


def _fit_linear(self, X, y, sample_weight=None):
    """
    Fit linear model.

    Parameters
    ----------
    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, n_targets]
        Target values

    sample_weight : numpy array of shape [n_samples]
        Individual weights for each sample

        .. versionadded:: 0.17
           parameter *sample_weight* support to LinearRegression.

    Returns
    -------
    self : returns an instance of self.
    """

    params = {
        'X': X,
        'y': y,
        'accept_sparse': ['csr', 'csc', 'coo'],
        'y_numeric': True,
        'multi_output': True,
    }
    if sklearn_check_version('0.23'):
        X, y = _daal_validate_data(
            self,
            dtype=[np.float64, np.float32],
            **params,
        )
    else:
        X, y = _daal_check_X_y(**params)

    dtype = get_dtype(X)

    self.fit_shape_good_for_daal_ = \
        bool(X.shape[0] > X.shape[1] + int(self.fit_intercept))

    daal_ready = self.fit_shape_good_for_daal_ and not sp.issparse(X) and \
        sample_weight is None
    if sklearn_check_version('0.22') and not sklearn_check_version('0.23'):
        daal_ready = daal_ready and dtype in [np.float32, np.float64]

    if daal_ready:
        logging.info(
            "sklearn.linar_model.LinearRegression."
            "fit: " + get_patch_message("daal"))
        res = _daal4py_fit(self, X, y)
        if res is not None:
            return res
        logging.info(
            "sklearn.linar_model.LinearRegression."
            "fit: " + get_patch_message("sklearn_after_daal"))
    else:
        logging.info(
            "sklearn.linar_model.LinearRegression."
            "fit: " + get_patch_message("sklearn"))

    return super(LinearRegression, self).fit(
        X,
        y,
        sample_weight=sample_weight,
    )


def _predict_linear(self, X):
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
    is_df = is_DataFrame(X)
    if sklearn_check_version('0.23'):
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
    X = np.asarray(X) if not sp.issparse(X) and not is_df else X
    good_shape_for_daal = \
        True if X.ndim <= 1 else True if X.shape[0] > X.shape[1] else False

    sklearn_ready = sp.issparse(X) or not hasattr(self, 'daal_model_') or \
        not self.fit_shape_good_for_daal_ or not good_shape_for_daal or \
        (hasattr(self, 'sample_weight') and self.sample_weight is not None)

    if sklearn_ready:
        logging.info(
            "sklearn.linar_model.LinearRegression."
            "predict: " + get_patch_message("sklearn"))
        return self._decision_function(X)
    logging.info(
        "sklearn.linar_model.LinearRegression."
        "predict: " + get_patch_message("daal"))
    X = _daal_check_array(X)
    return _daal4py_predict(self, X)


class LinearRegression(LinearRegression_original):
    __doc__ = LinearRegression_original.__doc__

    if sklearn_check_version('0.24'):
        def __init__(
            self,
            fit_intercept=True,
            normalize='deprecated' if sklearn_check_version('1.0') else False,
            copy_X=True,
            n_jobs=None,
            positive=False,
        ):
            super(LinearRegression, self).__init__(
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive,
            )
    else:
        def __init__(
            self,
            fit_intercept=True,
            normalize=False,
            copy_X=True,
            n_jobs=None,
        ):
            super(LinearRegression, self).__init__(
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs
            )

    @support_usm_ndarray()
    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version('1.0'):
            self._normalize = _deprecate_normalize(
                self.normalize,
                default=False,
                estimator_name=self.__class__.__name__,
            )

        if sklearn_check_version('0.24'):
            if self.positive is True:
                logging.info(
                    "sklearn.linar_model.LinearRegression."
                    "fit: " + get_patch_message("sklearn"))
                return super(LinearRegression, self).fit(
                    X, y=y, sample_weight=sample_weight
                )
        return _fit_linear(self, X, y, sample_weight=sample_weight)

    @support_usm_ndarray()
    def predict(self, X):
        return _predict_linear(self, X)
