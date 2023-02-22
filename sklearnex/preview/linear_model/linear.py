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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
import logging

if daal_check_version((2023, 'P', 100)):
    import numpy as np

    from ._common import BaseLinearRegression
    from ..._device_offload import dispatch, wrap_output_data

    from ...utils.validation import assert_all_finite
    from daal4py.sklearn._utils import (get_dtype, make2d)
    from sklearn.linear_model import LinearRegression as sklearn_LinearRegression

    if sklearn_check_version('1.0') and not sklearn_check_version('1.2'):
        from sklearn.linear_model._base import _deprecate_normalize

    from sklearn.utils.validation import _deprecate_positional_args
    from sklearn.exceptions import NotFittedError
    from scipy.sparse import issparse

    from onedal.linear_model import LinearRegression as onedal_LinearRegression
    from onedal.datatypes import (_num_samples, _get_2d_shape)

    class LinearRegression(sklearn_LinearRegression, BaseLinearRegression):
        __doc__ = sklearn_LinearRegression.__doc__
        intercept_, coef_ = None, None

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
            if sklearn_check_version('1.0'):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()

            dispatch(self, 'linear_model.LinearRegression.fit', {
                'onedal': self.__class__._onedal_fit,
                'sklearn': sklearn_LinearRegression.fit,
            }, X, y, sample_weight)
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

        def _test_type_and_finiteness(self, X_in):
            X = X_in if isinstance(X_in, np.ndarray) else np.asarray(X_in)

            dtype = X.dtype
            if 'complex' in str(type(dtype)):
                return False

            try:
                assert_all_finite(X)
            except BaseException:
                return False
            return True

        def _onedal_fit_supported(self, method_name, *data):
            assert method_name == 'linear_model.LinearRegression.fit'

            assert len(data) == 3
            X, y, sample_weight = data

            if sample_weight is not None:
                return False

            if issparse(X) or issparse(y):
                return False

            if hasattr(self, 'normalize') and self.normalize \
                    and self.normalize != 'deprecated':
                return False

            if hasattr(self, 'positive') and self.positive:
                return False

            n_samples, n_features = _get_2d_shape(X, fallback_1d=True)

            # Check if equations are well defined
            is_good_for_onedal = n_samples > \
                (n_features + int(self.fit_intercept))
            if not is_good_for_onedal:
                return False

            if not self._test_type_and_finiteness(X):
                return False

            if not self._test_type_and_finiteness(y):
                return False

            return True

        def _onedal_predict_supported(self, method_name, *data):
            assert method_name == 'linear_model.LinearRegression.predict'

            assert len(data) == 1

            n_samples = _num_samples(*data)
            if not (n_samples > 0):
                return False

            if issparse(*data) or issparse(self.coef_):
                return False

            if self.fit_intercept and issparse(self.intercept_):
                return False

            if not hasattr(self, '_onedal_estimator'):
                return False

            if not self._test_type_and_finiteness(*data):
                return False

            return True

        def _onedal_supported(self, method_name, *data):
            if method_name == 'linear_model.LinearRegression.fit':
                return self._onedal_fit_supported(method_name, *data)
            if method_name == 'linear_model.LinearRegression.predict':
                return self._onedal_predict_supported(method_name, *data)
            raise RuntimeError(
                f'Unknown method {method_name} in {self.__class__.__name__}')

        def _onedal_gpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _onedal_cpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _initialize_onedal_estimator(self):
            onedal_params = {
                'fit_intercept': self.fit_intercept,
                'copy_X': self.copy_X}
            self._onedal_estimator = onedal_LinearRegression(**onedal_params)

        def _onedal_fit(self, X, y, sample_weight, queue=None):
            assert sample_weight is None

            if sklearn_check_version(
                    '1.0') and not sklearn_check_version('1.2'):
                self._normalize = _deprecate_normalize(
                    self.normalize,
                    default=False,
                    estimator_name=self.__class__.__name__,
                )

            self._initialize_onedal_estimator()
            self._onedal_estimator.fit(X, y, queue=queue)

            self._save_attributes()

        def _onedal_predict(self, X, queue=None):
            if not hasattr(self, '_onedal_estimator'):
                self._initialize_onedal_estimator()
                self._onedal_estimator.coef_ = self.coef_
                self._onedal_estimator.intercept_ = self.intercept_

            return self._onedal_estimator.predict(X, queue=queue)

else:
    from daal4py.sklearn.linear_model import LinearRegression
    logging.warning('Preview LinearRegression requires oneDAL version >= 2023.1 '
                    'but it was not found')
