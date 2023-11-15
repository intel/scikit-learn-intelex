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

import logging
from abc import ABC

from daal4py.sklearn._utils import daal_check_version


def get_coef(self):
    return self._coef_


def set_coef(self, value):
    self._coef_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.coef_ = value
        if not self._is_in_fit:
            del self._onedal_estimator._onedal_model


def get_intercept(self):
    return self._intercept_


def set_intercept(self, value):
    self._intercept_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.intercept_ = value
        if not self._is_in_fit:
            del self._onedal_estimator._onedal_model


class BaseLinearRegression(ABC):
    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self._coef_ = self._onedal_estimator.coef_
        self._intercept_ = self._onedal_estimator.intercept_
        self._sparse = False

        self.coef_ = property(get_coef, set_coef)
        self.intercept_ = property(get_intercept, set_intercept)

        self._is_in_fit = True
        self.coef_ = self._coef_
        self.intercept_ = self._intercept_
        self._is_in_fit = False


if daal_check_version((2023, "P", 100)):
    import numpy as np
    from sklearn.linear_model import LinearRegression as sklearn_LinearRegression

    from daal4py.sklearn._utils import get_dtype, make2d, sklearn_check_version

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain, get_patch_message
    from ..utils.validation import _assert_all_finite

    if sklearn_check_version("1.0") and not sklearn_check_version("1.2"):
        from sklearn.linear_model._base import _deprecate_normalize

    from scipy.sparse import issparse
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import _deprecate_positional_args, check_X_y

    from onedal.linear_model import LinearRegression as onedal_LinearRegression
    from onedal.utils import _num_features, _num_samples

    class LinearRegression(sklearn_LinearRegression, BaseLinearRegression):
        __doc__ = sklearn_LinearRegression.__doc__
        intercept_, coef_ = None, None

        if sklearn_check_version("1.2"):
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

        elif sklearn_check_version("0.24"):

            def __init__(
                self,
                fit_intercept=True,
                normalize="deprecated" if sklearn_check_version("1.0") else False,
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
                    n_jobs=n_jobs,
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
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()

            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": sklearn_LinearRegression.fit,
                },
                X,
                y,
                sample_weight,
            )
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
            return dispatch(
                self,
                "predict",
                {
                    "onedal": self.__class__._onedal_predict,
                    "sklearn": sklearn_LinearRegression.predict,
                },
                X,
            )

        def _test_type_and_finiteness(self, X_in):
            X = X_in if isinstance(X_in, np.ndarray) else np.asarray(X_in)

            dtype = X.dtype
            if "complex" in str(type(dtype)):
                return False

            try:
                _assert_all_finite(X)
            except BaseException:
                return False
            return True

        def _onedal_fit_supported(self, method_name, *data):
            assert method_name == "fit"
            assert len(data) == 3
            X, y, sample_weight = data

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.fit"
            )

            normalize_is_set = (
                hasattr(self, "normalize")
                and self.normalize
                and self.normalize != "deprecated"
            )
            positive_is_set = hasattr(self, "positive") and self.positive

            n_samples = _num_samples(X)
            n_features = _num_features(X, fallback_1d=True)

            # Check if equations are well defined
            is_good_for_onedal = n_samples > (n_features + int(self.fit_intercept))

            dal_ready = patching_status.and_conditions(
                [
                    (sample_weight is None, "Sample weight is not supported."),
                    (
                        not issparse(X) and not issparse(y),
                        "Sparse input is not supported.",
                    ),
                    (not normalize_is_set, "Normalization is not supported."),
                    (
                        not positive_is_set,
                        "Forced positive coefficients are not supported.",
                    ),
                    (
                        is_good_for_onedal,
                        "The shape of X (fitting) does not satisfy oneDAL requirements:."
                        "Number of features + 1 >= number of samples.",
                    ),
                ]
            )
            if not dal_ready:
                return patching_status

            if not patching_status.and_condition(
                self._test_type_and_finiteness(X), "Input X is not supported."
            ):
                return patching_status

            patching_status.and_condition(
                self._test_type_and_finiteness(y), "Input y is not supported."
            )

            return patching_status

        def _onedal_predict_supported(self, method_name, *data):
            assert method_name == "predict"
            assert len(data) == 1

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.predict"
            )

            n_samples = _num_samples(*data)
            model_is_sparse = issparse(self.coef_) or (
                self.fit_intercept and issparse(self.intercept_)
            )
            dal_ready = patching_status.and_conditions(
                [
                    (n_samples > 0, "Number of samples is less than 1."),
                    (not issparse(*data), "Sparse input is not supported."),
                    (not model_is_sparse, "Sparse coefficients are not supported."),
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."),
                ]
            )
            if not dal_ready:
                return patching_status

            patching_status.and_condition(
                self._test_type_and_finiteness(*data), "Input X is not supported."
            )

            return patching_status

        def _onedal_supported(self, method_name, *data):
            if method_name == "fit":
                return self._onedal_fit_supported(method_name, *data)
            if method_name == "predict":
                return self._onedal_predict_supported(method_name, *data)
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_gpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _onedal_cpu_supported(self, method_name, *data):
            return self._onedal_supported(method_name, *data)

        def _initialize_onedal_estimator(self):
            onedal_params = {"fit_intercept": self.fit_intercept, "copy_X": self.copy_X}
            self._onedal_estimator = onedal_LinearRegression(**onedal_params)

        def _onedal_fit(self, X, y, sample_weight, queue=None):
            assert sample_weight is None

            check_params = {
                "X": X,
                "y": y,
                "dtype": [np.float64, np.float32],
                "accept_sparse": ["csr", "csc", "coo"],
                "y_numeric": True,
                "multi_output": True,
                "force_all_finite": False,
            }
            if sklearn_check_version("1.2"):
                X, y = self._validate_data(**check_params)
            else:
                X, y = check_X_y(**check_params)

            if sklearn_check_version("1.0") and not sklearn_check_version("1.2"):
                self._normalize = _deprecate_normalize(
                    self.normalize,
                    default=False,
                    estimator_name=self.__class__.__name__,
                )

            self._initialize_onedal_estimator()
            try:
                self._onedal_estimator.fit(X, y, queue=queue)
                self._save_attributes()

            except RuntimeError:
                logging.getLogger("sklearnex").info(
                    f"{self.__class__.__name__}.fit "
                    + get_patch_message("sklearn_after_onedal")
                )

                del self._onedal_estimator
                super().fit(X, y)

        def _onedal_predict(self, X, queue=None):
            X = self._validate_data(X, accept_sparse=False, reset=False)
            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.coef_ = self.coef_
                self._onedal_estimator.intercept_ = self.intercept_

            return self._onedal_estimator.predict(X, queue=queue)

else:
    from daal4py.sklearn.linear_model import LinearRegression

    logging.warning(
        "Sklearnex LinearRegression requires oneDAL version >= 2023.1 "
        "but it was not found"
    )
