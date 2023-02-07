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
from ._common import BaseLinearRegression
from .._device_offload import dispatch, wrap_output_data

from sklearn.linear_model import LinearRegression as sklearn_LinearRegression
if sklearn_check_version('1.0') and not sklearn_check_version('1.2'):
    from sklearn.linear_model._base import _deprecate_normalize

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.exceptions import NotFittedError
from scipy import sparse as sp
from numbers import Integral

from onedal.linear_model import LinearRegression as onedal_LinearRegression


class LinearRegression(sklearn_LinearRegression, BaseLinearRegression):
    __doc__ = sklearn_LinearRegression.__doc__

    if sklearn_check_version('1.2'):
        _parameter_constraints: dict = {
            **sklearn_LinearRegression._parameter_constraints
        }

        def __init__(
            self,
            fit_intercept=True,
            copy_X=True,
            n_jobs=None,
            positive=False,
        ):
            super().__init__(
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                n_jobs=n_jobs,
                positive=positive,
            )
    elif sklearn_check_version('0.24'):
        def __init__(
            self,
            fit_intercept=True,
            normalize='deprecated' if sklearn_check_version('1.0') else False,
            copy_X=True,
            n_jobs=None,
            positive=False,
        ):
            super().__init__(
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
            super().__init__(
                fit_intercept=fit_intercept,
                normalize=normalize,
                copy_X=copy_X,
                n_jobs=n_jobs
            )

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        if sklearn_check_version('1.0') and not sklearn_check_version('1.2'):
            self._normalize = _deprecate_normalize(
                self.normalize,
                default=False,
                estimator_name=self.__class__.__name__,
            )
        if sklearn_check_version('1.0'):
            self._check_feature_names(X, reset=True)
        if sklearn_check_version("1.2"):
            self._validate_params()

        dispatch(self, 'linear_model.LinearRegression.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_LinearRegression.fit,
        }, X, y)
        return self

    @wrap_output_data
    def predict(self, X):
        """
        Predict using the linear model.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape (n_samples, n_targets)
            Returns predicted values.
        """
        if sklearn_check_version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'linear_model.LinearRegression.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_LinearRegression.predict,
        }, X)

    def _onedal_supported(self, method_name, *data):
        if method_name in ['linear_model.LinearRegression.fit',
                           'linear_model.LinearRegression.predict']:
            if hasattr(self, 'normalize') and self.normalize:
                return False
            if hasattr(self, 'positive') and self.positive:
                return False
            if len(data) > 1:
                self._is_sparse = sp.isspmatrix(data[0])
            return hasattr(self, '_is_sparse') and not self._is_sparse
        raise RuntimeError(
            f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_fit(self, X, y, queue=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        onedal_params = {'fit_intercept': self.fit_intercept}

        self._onedal_estimator = onedal_LinearRegression(**onedal_params)
        self._onedal_estimator.fit(X, y, queue=queue)

        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)
