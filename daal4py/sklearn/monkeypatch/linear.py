import numpy as np
from scipy import sparse as sp
from scipy import linalg
from sklearn.utils import check_array, check_X_y, deprecated, as_float_array
from sklearn.linear_model.base import _rescale_data
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import sparse_lsqr

import daal4py
from ..utils import (make2d, getFPType)

def _daal4py_fit(self, X, y):
    y = make2d(y)
    X_fptype = getFPType(X)
    lr_algorithm = daal4py.linear_regression_training(
        fptype=X_fptype,
        interceptFlag=bool(self.fit_intercept),
        method='defaultDense')

    lr_res = lr_algorithm.compute(X, y)
    lr_model = lr_res.model
    self.daal_model_ = lr_model
    coefs = lr_model.Beta

    self.intercept_ = coefs[:,0].copy(order='C')
    self.coef_ = coefs[:,1:].copy(order='C')

    if self.coef_.shape[0] == 1:
        self.coef_ = np.ravel(self.coef_)
        self.intercept_ = self.intercept_[0]

    return self


def _daal4py_predict(self, X):
    X = make2d(X)
    _fptype = getFPType(self.coef_)
    lr_pred = daal4py.linear_regression_prediction(
        fptype=_fptype,
        method='defaultDense')
    lr_res = lr_pred.compute(X, self.daal_model_)
    res = lr_res.prediction
    if res.shape[1] == 1:
        res = np.ravel(res)
    return res


def fit(self, X, y, sample_weight=None):
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

    n_jobs_ = self.n_jobs
    X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                     y_numeric=True, multi_output=True)

    self.sample_weight_ = sample_weight
    self.fit_shape_good_for_daal_ = bool(X.shape[0] > X.shape[1] + int(self.fit_intercept))
    if (self.fit_shape_good_for_daal_ and
            not sp.issparse(X) and
            (X.dtype == np.float64 or X.dtype == np.float32) and
            sample_weight is None):
        _daal4py_fit(self, X, y)
        return self

    if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
        raise ValueError("Sample weights must be 1D array or scalar")

    X, y, X_offset, y_offset, X_scale = self._preprocess_data(
        X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
        copy=self.copy_X, sample_weight=sample_weight)

    if sample_weight is not None:
        # Sample weight can be implemented via a simple rescaling.
        X, y = _rescale_data(X, y, sample_weight)

    if sp.issparse(X):
        if y.ndim < 2:
            out = sparse_lsqr(X, y)
            self.coef_ = out[0]
            self._residues = out[3]
        else:
            # sparse_lstsq cannot handle y with shape (M, K)
            outs = Parallel(n_jobs=n_jobs_)(
                delayed(sparse_lsqr)(X, y[:, j].ravel())
                for j in range(y.shape[1]))
            self.coef_ = np.vstack(out[0] for out in outs)
            self._residues = np.vstack(out[3] for out in outs)
    else:
        self.coef_, self._residues, self.rank_, self.singular_ = \
            linalg.lstsq(X, y)
        self.coef_ = self.coef_.T

    if y.ndim == 1:
        self.coef_ = np.ravel(self.coef_)
    self._set_intercept(X_offset, y_offset, X_scale)
    return self

def predict(self, X):
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
    X = np.asarray(X) if not sp.issparse(X) else X
    good_shape_for_daal = True if X.ndim <= 1 else True if X.shape[0] > X.shape[1] else False

    if (sp.issparse(X) or
            not hasattr(self, 'daal_model_') or
            not self.fit_shape_good_for_daal_ or
            not good_shape_for_daal or
            not (X.dtype == np.float64 or X.dtype == np.float32) or
            (hasattr(self, 'sample_weight_') and self.sample_weight_ is not None)):
        return self._decision_function(X)
    else:
        X = check_array(X)
        return _daal4py_predict(self, X)
