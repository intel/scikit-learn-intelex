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

import sklearn.linear_model._logistic as logistic_module

from daal4py.sklearn._utils import daal_check_version
from daal4py.sklearn.linear_model.logistic_path import (
    LogisticRegression,
    daal4py_predict,
    logistic_regression_path,
)


class BaseLogisticRegression(ABC):
    def _save_attributes(self):
        assert hasattr(self, "_onedal_estimator")
        self.classes_ = self._onedal_estimator.classes_
        self.coef_ = self._onedal_estimator.coef_
        self.intercept_ = self._onedal_estimator.intercept_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_iter_ = self._onedal_estimator.n_iter_


if daal_check_version((2024, "P", 1)):
    import numpy as np
    from scipy.sparse import issparse
    from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
    from sklearn.utils.validation import check_X_y

    from daal4py.sklearn._utils import sklearn_check_version
    from onedal.linear_model import LogisticRegression as onedal_LogisticRegression
    from onedal.utils import _num_features, _num_samples

    from ..._device_offload import dispatch, wrap_output_data
    from ..._utils import PatchingConditionsChain, get_patch_message
    from ...utils.validation import _assert_all_finite

    class LogisticRegression(sklearn_LogisticRegression, BaseLogisticRegression):
        __doc__ = sklearn_LogisticRegression.__doc__
        intercept_, coef_, n_iter_ = None, None, None

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {
                **sklearn_LogisticRegression._parameter_constraints
            }

        def __init__(
            self,
            penalty="l2",
            *,
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver="lbfgs" if sklearn_check_version("0.22") else "liblinear",
            max_iter=100,
            multi_class="auto" if sklearn_check_version("0.22") else "ovr",
            verbose=0,
            warm_start=False,
            n_jobs=None,
            l1_ratio=None,
        ):
            super().__init__(
                penalty=penalty,
                dual=dual,
                tol=tol,
                C=C,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                random_state=random_state,
                solver=solver,
                max_iter=max_iter,
                multi_class=multi_class,
                verbose=verbose,
                warm_start=warm_start,
                n_jobs=n_jobs,
                l1_ratio=l1_ratio,
            )

        def fit(self, X, y, sample_weight=None):
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=True)
            if sklearn_check_version("1.2"):
                self._validate_params()
            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": sklearn_LogisticRegression.fit,
                },
                X,
                y,
                sample_weight,
            )
            return self

        @wrap_output_data
        def predict(self, X):
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=False)
            return dispatch(
                self,
                "predict",
                {
                    "onedal": self.__class__._onedal_predict,
                    "sklearn": sklearn_LogisticRegression.predict,
                },
                X,
            )

        @wrap_output_data
        def predict_proba(self, X):
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=False)
            return dispatch(
                self,
                "predict",
                {
                    "onedal": self.__class__._onedal_predict_proba,
                    "sklearn": sklearn_LogisticRegression.predict_proba,
                },
                X,
            )

        @wrap_output_data
        def predict_log_proba(self, X):
            if sklearn_check_version("1.0"):
                self._check_feature_names(X, reset=False)
            return dispatch(
                self,
                "predict",
                {
                    "onedal": self.__class__._onedal_predict_log_proba,
                    "sklearn": sklearn_LogisticRegression.predict_log_proba,
                },
                X,
            )

        def _test_type_and_finiteness(self, X_in):
            X = np.asarray(X_in)

            dtype = X.dtype
            if "complex" in str(type(dtype)):
                return False

            try:
                _assert_all_finite(X)
            except BaseException:
                return False
            return True

        def _onedal_gpu_fit_supported(self, method_name, *data):
            assert method_name == "fit"
            assert len(data) == 3
            X, y, sample_weight = data

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.fit"
            )

            dal_ready = patching_status.and_conditions(
                [
                    (self.penalty == "l2", "Only l2 penalty is supported."),
                    (self.dual == False, "dual=True is not supported."),
                    (self.intercept_scaling == 1, "Intercept scaling is not supported."),
                    (self.class_weight is None, "Class weight is not supported"),
                    (self.solver == "newton-cg", "Only newton-cg solver is supported."),
                    (
                        self.multi_class != "multinomial",
                        "multi_class parameter is not supported.",
                    ),
                    (self.warm_start == False, "Warm start is not supported."),
                    (self.l1_ratio is None, "l1 ratio is not supported."),
                    (sample_weight is None, "Sample weight is not supported."),
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

        def _onedal_gpu_predict_supported(self, method_name, *data):
            assert method_name in ["predict", "predict_proba", "predict_log_proba"]
            assert len(data) == 1

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.{method_name}"
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

        def _onedal_gpu_supported(self, method_name, *data):
            if method_name == "fit":
                return self._onedal_gpu_fit_supported(method_name, *data)
            if method_name in ["predict", "predict_proba", "predict_log_proba"]:
                return self._onedal_gpu_predict_supported(method_name, *data)
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        def _onedal_cpu_supported(self, method_name, *data):
            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.{method_name}"
            )

            return patching_status

        def _initialize_onedal_estimator(self):
            onedal_params = {
                "tol": self.tol,
                "C": self.C,
                "fit_intercept": self.fit_intercept,
                "solver": self.solver,
                "max_iter": self.max_iter,
            }
            self._onedal_estimator = onedal_LogisticRegression(**onedal_params)

        def _onedal_cpu_fit(self, X, y, sample_weight):
            which, what = logistic_module, "_logistic_regression_path"
            replacer = logistic_regression_path
            descriptor = getattr(which, what, None)
            setattr(which, what, replacer)
            clf = super().fit(X, y, sample_weight)
            setattr(which, what, descriptor)
            return clf

        def _onedal_fit(self, X, y, sample_weight, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return self._onedal_cpu_fit(X, y, sample_weight)

            assert sample_weight is None

            check_params = {
                "X": X,
                "y": y,
                "dtype": [np.float64, np.float32],
                "accept_sparse": False,
                "multi_output": False,
                "force_all_finite": True,
            }
            if sklearn_check_version("1.2"):
                X, y = self._validate_data(**check_params)
            else:
                X, y = check_X_y(**check_params)
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
            if queue is None or queue.sycl_device.is_cpu:
                return daal4py_predict(self, X, "computeClassLabels")

            X = self._validate_data(X, accept_sparse=False, reset=False)
            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.coef_ = self.coef_
                self._onedal_estimator.intercept_ = self.intercept_
                self._onedal_estimator.classes_ = self.classes_

            return self._onedal_estimator.predict(X, queue=queue)

        def _onedal_predict_proba(self, X, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return daal4py_predict(self, X, "computeClassProbabilities")
            X = self._validate_data(X, accept_sparse=False, reset=False)
            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.coef_ = self.coef_
                self._onedal_estimator.intercept_ = self.intercept_

            return self._onedal_estimator.predict_proba(X, queue=queue)

        def _onedal_predict_log_proba(self, X, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return daal4py_predict(self, X, "computeClassLogProbabilities")
            X = self._validate_data(X, accept_sparse=False, reset=False)
            if not hasattr(self, "_onedal_estimator"):
                self._initialize_onedal_estimator()
                self._onedal_estimator.coef_ = self.coef_
                self._onedal_estimator.intercept_ = self.intercept_

            return self._onedal_estimator.predict_log_proba(X, queue=queue)

else:
    from daal4py.sklearn.linear_model import LogisticRegression

    logging.warning(
        "Sklearnex LogisticRegression requires oneDAL version >= 2024.0.1 "
        "but it was not found"
    )
