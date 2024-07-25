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

if daal_check_version((2024, "P", 600)):
    import numbers

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.linear_model import Ridge as sklearn_Ridge
    from sklearn.metrics import r2_score
    from sklearn.utils.validation import check_is_fitted, check_X_y

    from daal4py.sklearn.linear_model._ridge import _fit_ridge as daal4py_fit_ridge

    if sklearn_check_version("1.0") and not sklearn_check_version("1.2"):
        from sklearn.linear_model._base import _deprecate_normalize
    if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
        from sklearn.utils import check_scalar

    from onedal.linear_model import Ridge as onedal_Ridge
    from onedal.utils import _num_features, _num_samples

    from ..._device_offload import dispatch, wrap_output_data
    from ..._utils import PatchingConditionsChain

    def _is_numeric_scalar(value):
        """
        Determines if the provided value is a numeric scalar.

        Args:
        value: The value to be checked.

        Returns:
        bool: True if the value is a numeric scalar, False otherwise.
        """
        return isinstance(value, numbers.Real)

    class Ridge(sklearn_Ridge):
        __doc__ = sklearn_Ridge.__doc__

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {**sklearn_Ridge._parameter_constraints}

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

        elif sklearn_check_version("1.0"):

            def __init__(
                self,
                alpha=1.0,
                fit_intercept=True,
                normalize="deprecated",
                copy_X=True,
                max_iter=None,
                tol=1e-3,
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

        else:

            def __init__(
                self,
                alpha=1.0,
                fit_intercept=True,
                normalize=False,
                copy_X=True,
                max_iter=None,
                tol=1e-3,
                solver="auto",
                random_state=None,
            ):
                super().__init__(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    normalize=normalize,
                    copy_X=copy_X,
                    max_iter=max_iter,
                    tol=tol,
                    solver=solver,
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
                    "sklearn": sklearn_Ridge.fit,
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
                    "sklearn": sklearn_Ridge.predict,
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
                    "sklearn": sklearn_Ridge.score,
                },
                X,
                y,
                sample_weight=sample_weight,
            )

        def _onedal_fit_supported(self, patching_status, method_name, *data):
            assert method_name == "fit"
            assert len(data) == 3
            X, y, sample_weight = data

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

            patching_status.and_conditions(
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

            return patching_status

        def _onedal_predict_supported(self, patching_status, method_name, *data):
            assert method_name in ["predict", "score"]
            assert len(data) <= 2

            n_samples = _num_samples(data[0])
            model_is_sparse = issparse(self.coef_) or (
                self.fit_intercept and issparse(self.intercept_)
            )
            patching_status.and_conditions(
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

            return patching_status

        def _onedal_gpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{self.__class__.__name__}.fit"
            )

            if method_name == "fit":
                patching_status.and_condition(
                    _is_numeric_scalar(self.alpha),
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
            self._onedal_estimator = onedal_Ridge(**onedal_params)

        def _daal_fit(self, X, y, sample_weight=None):
            daal4py_fit_ridge(self, X, y, sample_weight)
            self._onedal_estimator.n_features_in_ = _num_features(X, fallback_1d=True)
            self._onedal_estimator.coef_ = self.coef_
            self._onedal_estimator.intercept_ = self.intercept_

        def _onedal_fit(self, X, y, sample_weight, queue=None):
            # `Sample weight` is not supported. Expected to be None value.
            assert sample_weight is None

            if sklearn_check_version("1.2"):
                self._validate_params()
            elif sklearn_check_version("1.1"):
                if self.max_iter is not None:
                    self.max_iter = check_scalar(
                        self.max_iter, "max_iter", target_type=numbers.Integral, min_val=1
                    )
                self.tol = check_scalar(
                    self.tol, "tol", target_type=numbers.Real, min_val=0.0
                )
                if self.alpha is not None and not isinstance(
                    self.alpha, (np.ndarray, tuple)
                ):
                    self.alpha = check_scalar(
                        self.alpha,
                        "alpha",
                        target_type=numbers.Real,
                        min_val=0.0,
                        include_boundaries="left",
                    )

            check_params = {
                "X": X,
                "y": y,
                "dtype": [np.float64, np.float32],
                "accept_sparse": ["csr", "csc", "coo"],
                "y_numeric": True,
                "multi_output": True,
            }
            if sklearn_check_version("1.0"):
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

            # Falling back to daal4py if the device is CPU and alpha is array-like
            # since onedal does not yet support non-scalars for alpha, thus
            # should only be used for GPU/CPU with scalar alpha to not limit the functionality
            cpu_device = queue is None or queue.sycl_device.is_cpu
            if cpu_device and not _is_numeric_scalar(self.alpha):
                self._daal_fit(X, y)
            else:
                self._onedal_estimator.fit(X, y, queue=queue)

            self._save_attributes()

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
            self._coef = self._onedal_estimator.coef_
            self._intercept = self._onedal_estimator.intercept_

        fit.__doc__ = sklearn_Ridge.fit.__doc__
        predict.__doc__ = sklearn_Ridge.predict.__doc__
        score.__doc__ = sklearn_Ridge.score.__doc__

else:
    from daal4py.sklearn.linear_model._ridge import Ridge
    from onedal._device_offload import support_usm_ndarray

    Ridge.fit = support_usm_ndarray(queue_param=False)(Ridge.fit)
    Ridge.predict = support_usm_ndarray(queue_param=False)(Ridge.predict)
    Ridge.score = support_usm_ndarray(queue_param=False)(Ridge.score)

    logging.warning(
        "Preview Ridge requires oneDAL version >= 2024.6 but it was not found"
    )
