# ===============================================================================
# Copyright 2024 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

if daal_check_version((2024, "P", 500)):
    from abc import ABC

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.exceptions import NotFittedError
    from sklearn.linear_model import Ridge as sklearn_RidgeRegression
    from sklearn.metrics import r2_score
    from sklearn.utils.validation import check_is_fitted, check_X_y

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn.linear_model._ridge import _fit_ridge as daal4py_fit_ridge

    if sklearn_check_version("1.0") and not sklearn_check_version("1.2"):
        from sklearn.linear_model._base import _deprecate_normalize

    from onedal.common.hyperparameters import get_hyperparameters
    from onedal.linear_model import LinearRegression as onedal_RidgeRegression
    from onedal.utils import _num_features, _num_samples

    from ..._device_offload import dispatch, wrap_output_data
    from ..._utils import (
        PatchingConditionsChain,
        get_patch_message,
        register_hyperparameters,
    )
    from ...utils import get_namespace
    from ...utils.validation import _assert_all_finite

    def is_numeric_scalar(value):
        return isinstance(value, (int, float))

    class Ridge(sklearn_RidgeRegression):
        __doc__ = sklearn_RidgeRegression.__doc__

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {
                **sklearn_RidgeRegression._parameter_constraints
            }

            def __init__(
                self,
                alpha=1.0,
                fit_intercept=True,
                copy_X=True,
                max_iter=None,
                tol=1e-4,
                solver="auto",
                positive=False,
                random_state=None,
            ):
                super().__init__(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    copy_X=copy_X,
                    max_iter=max_iter,
                    tol=tol,
                    solver=solver,
                    positive=positive,
                    random_state=random_state,
                )

        else:

            def __init__(
                self,
                alpha=1.0,
                fit_intercept=True,
                normalize="deprecated" if sklearn_check_version("1.0") else False,
                copy_X=True,
                max_iter=None,
                tol=1e-4,
                solver="auto",
                positive=False,
                random_state=None,
            ):
                super().__init__(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    normalize=normalize,
                    copy_X=copy_X,
                    max_iter=max_iter,
                    solver=solver,
                    tol=tol,
                    positive=positive,
                    random_state=random_state,
                )

        def fit(self, X, y, sample_weight=None):
            # It is necessary to properly update coefs for predict if we
            # fallback to sklearn in dispatch
            if hasattr(self, "_onedal_estimator"):
                del self._onedal_estimator

            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": sklearn_RidgeRegression.fit,
                },
                X,
                y,
                sample_weight,
            )
            return self

        @wrap_output_data
        def predict(self, X):
            check_is_fitted(self)

            return dispatch(
                self,
                "predict",
                {
                    "onedal": self.__class__._onedal_predict,
                    "sklearn": sklearn_RidgeRegression.predict,
                },
                X,
            )

        @wrap_output_data
        def score(self, X, y, sample_weight=None):
            check_is_fitted(self)

            return dispatch(
                self,
                "score",
                {
                    "onedal": self.__class__._onedal_score,
                    "sklearn": sklearn_RidgeRegression.score,
                },
                X,
                y,
                sample_weight=sample_weight,
            )

        def _test_type_and_finiteness(self, X_in):
            xp, _ = get_namespace(X_in)
            X = xp.asarray(X_in)

            if np.iscomplexobj(X):
                return False

            try:
                _assert_all_finite(X)
            except BaseException:
                return False

            return True

        def _onedal_fit_supported(self, patching_status, method_name, *data):
            assert method_name == "fit"
            assert len(data) == 3
            X, y, sample_weight = data

            if not patching_status:
                patching_status = PatchingConditionsChain(
                    f"sklearn.linear_model.{self.__class__.__name__}.fit"
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
            is_underdetermined = n_samples < (n_features + int(self.fit_intercept))

            dal_ready = patching_status.and_conditions(
                [
                    (
                        self.solver == "auto",
                        f"'{self.solver}' solver is not supported. "
                        "Only 'auto' solver is supported.",
                    ),
                    (
                        not issparse(X) and not issparse(y),
                        "Sparse input is not supported.",
                    ),
                    (
                        not is_underdetermined,
                        "The shape of X (fitting) does not satisfy oneDAL requirements:"
                        "Number of features + 1 >= number of samples.",
                    ),
                    (sample_weight is None, "Sample weight is not supported."),
                    (not normalize_is_set, "Normalization is not supported."),
                    (
                        not positive_is_set,
                        "Forced positive coefficients are not supported.",
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

        def _onedal_predict_supported(self, patching_status, method_name, *data):
            assert method_name in ["predict", "score"]
            assert len(data) <= 2

            if not patching_status:
                patching_status = PatchingConditionsChain(
                    f"sklearn.linear_model.{self.__class__.__name__}.fit"
                )

            n_samples = _num_samples(data[0])
            model_is_sparse = issparse(self.coef_) or (
                self.fit_intercept and issparse(self.intercept_)
            )
            dal_ready = patching_status.and_conditions(
                [
                    (
                        self.solver == "auto",
                        f"'{self.solver}' solver is not supported. "
                        "Only 'auto' solver is supported.",
                    ),
                    (n_samples > 0, "Number of samples is less than 1."),
                    (not issparse(data[0]), "Sparse input is not supported."),
                    (not model_is_sparse, "Sparse coefficients are not supported."),
                ]
            )
            if not dal_ready:
                return patching_status

            patching_status.and_condition(
                self._test_type_and_finiteness(data[0]), "Input X is not supported."
            )

            return patching_status

        def _onedal_gpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{self.__class__.__name__}.fit"
            )

            if method_name == "fit":
                alpha_is_scalar = is_numeric_scalar(self.alpha)
                dal_ready = patching_status.and_condition(
                    alpha_is_scalar,
                    "Non-scalar alpha is not supported for GPU.",
                )

                return self._onedal_fit_supported(patching_status, method_name, *data)

            if method_name in ["predict", "score"]:
                return self._onedal_predict_supported(patching_status, method_name, *data)

            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_cpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{self.__class__.__name__}.fit"
            )

            if method_name == "fit":
                return self._onedal_fit_supported(patching_status, method_name, *data)

            if method_name in ["predict", "score"]:
                return self._onedal_predict_supported(patching_status, method_name, *data)

            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _initialize_onedal_estimator(self):
            onedal_params = {
                "fit_intercept": self.fit_intercept,
                "alpha": self.alpha,
                "copy_X": self.copy_X,
            }
            self._onedal_estimator = onedal_RidgeRegression(**onedal_params)

        def _daal_fit(self, X, y, sample_weight=None):
            daal4py_fit_ridge(self, X, y, sample_weight)
            self._onedal_estimator.n_features_in_ = _num_features(X, fallback_1d=True)
            self._onedal_estimator.coef_ = self.coef_
            self._onedal_estimator.intercept_ = self.intercept_

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
                self._validate_params()
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
                # Falling back to daal4py if the device is CPU since
                # onedal does not support non-scalars for alpha, thus
                # should only be used for GPU/CPU with scalar alpha to not limit the functionality
                cpu_device = queue is None or queue.sycl_device.is_cpu
                if cpu_device and not is_numeric_scalar(self.alpha):
                    self._daal_fit(X, y)
                else:
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
            if sklearn_check_version("1.0"):
                X = self._validate_data(X, accept_sparse=False, reset=False)

            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.coef_ = self.coef_
                self._onedal_estimator.intercept_ = self.intercept_

            res = self._onedal_estimator.predict(X, queue=queue)
            return res

        def _onedal_score(self, X, y, sample_weight=None, queue=None):
            return r2_score(
                y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
            )

        @property
        def coef_(self):
            return self._coef

        @coef_.setter
        def coef_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.coef_ = value
                # checking if the model is already fitted and if so, deleting the model
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._coef = value

        @property
        def intercept_(self):
            return self._intercept

        @intercept_.setter
        def intercept_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.intercept_ = value
                # checking if the model is already fitted and if so, deleting the model
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._intercept = value

        def _save_attributes(self):
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            self._sparse = False
            self._coef = self._onedal_estimator.coef_
            self._intercept = self._onedal_estimator.intercept_

        fit.__doc__ = sklearn_RidgeRegression.fit.__doc__
        predict.__doc__ = sklearn_RidgeRegression.predict.__doc__
        score.__doc__ = sklearn_RidgeRegression.score.__doc__

else:
    from daal4py.sklearn.linear_model._ridge import Ridge

    logging.warning(
        "Preview Ridge requires oneDAL version >= 2024.5 but it was not found"
    )
