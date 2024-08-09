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
from abc import ABC

from daal4py.sklearn._utils import daal_check_version
from daal4py.sklearn.linear_model.logistic_path import (
    LogisticRegression as LogisticRegression_daal4py,
)

if daal_check_version((2024, "P", 1)):
    import numpy as np
    from scipy.sparse import issparse
    from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.utils.multiclass import type_of_target
    from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import sklearn_check_version
    from daal4py.sklearn.linear_model.logistic_path import daal4py_fit, daal4py_predict
    from onedal.linear_model import LogisticRegression as onedal_LogisticRegression
    from onedal.utils import _num_samples

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain, get_patch_message

    class BaseLogisticRegression(ABC):
        def _save_attributes(self):
            assert hasattr(self, "_onedal_estimator")
            self.classes_ = self._onedal_estimator.classes_
            self.coef_ = self._onedal_estimator.coef_
            self.intercept_ = self._onedal_estimator.intercept_
            self.n_features_in_ = self._onedal_estimator.n_features_in_
            self.n_iter_ = self._onedal_estimator.n_iter_

    @control_n_jobs(
        decorated_methods=[
            "fit",
            "predict",
            "predict_proba",
            "predict_log_proba",
            "score",
        ]
    )
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
            solver="lbfgs",
            max_iter=100,
            multi_class="deprecated" if sklearn_check_version("1.5") else "auto",
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

        _onedal_cpu_fit = daal4py_fit

        def fit(self, X, y, sample_weight=None):
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
            return dispatch(
                self,
                "predict_proba",
                {
                    "onedal": self.__class__._onedal_predict_proba,
                    "sklearn": sklearn_LogisticRegression.predict_proba,
                },
                X,
            )

        @wrap_output_data
        def predict_log_proba(self, X):
            return dispatch(
                self,
                "predict_log_proba",
                {
                    "onedal": self.__class__._onedal_predict_log_proba,
                    "sklearn": sklearn_LogisticRegression.predict_log_proba,
                },
                X,
            )

        @wrap_output_data
        def score(self, X, y, sample_weight=None):
            return dispatch(
                self,
                "score",
                {
                    "onedal": self.__class__._onedal_score,
                    "sklearn": sklearn_LogisticRegression.score,
                },
                X,
                y,
                sample_weight=sample_weight,
            )

        def _onedal_score(self, X, y, sample_weight=None, queue=None):
            return accuracy_score(
                y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
            )

        def _onedal_gpu_fit_supported(self, method_name, *data):
            assert method_name == "fit"
            assert len(data) == 3
            X, y, sample_weight = data

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.fit"
            )

            target_type = (
                type_of_target(y, input_name="y")
                if sklearn_check_version("1.1")
                else type_of_target(y)
            )
            patching_status.and_conditions(
                [
                    (self.penalty == "l2", "Only l2 penalty is supported."),
                    (self.dual == False, "dual=True is not supported."),
                    (
                        self.intercept_scaling == 1,
                        "Intercept scaling is not supported.",
                    ),
                    (self.class_weight is None, "Class weight is not supported"),
                    (self.solver == "newton-cg", "Only newton-cg solver is supported."),
                    (
                        self.multi_class != "multinomial",
                        "multi_class parameter is not supported.",
                    ),
                    (self.warm_start == False, "Warm start is not supported."),
                    (self.l1_ratio is None, "l1 ratio is not supported."),
                    (sample_weight is None, "Sample weight is not supported."),
                    (
                        target_type == "binary",
                        "Only binary classification is supported",
                    ),
                ]
            )

            return patching_status

        def _onedal_gpu_predict_supported(self, method_name, *data):
            assert method_name in [
                "predict",
                "predict_proba",
                "predict_log_proba",
                "score",
            ]

            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.linear_model.{class_name}.{method_name}"
            )

            n_samples = _num_samples(data[0])
            model_is_sparse = issparse(self.coef_) or (
                self.fit_intercept and issparse(self.intercept_)
            )
            dal_ready = patching_status.and_conditions(
                [
                    (n_samples > 0, "Number of samples is less than 1."),
                    (
                        not any([issparse(i) for i in data]),
                        "Sparse input is not supported.",
                    ),
                    (not model_is_sparse, "Sparse coefficients are not supported."),
                    (
                        hasattr(self, "_onedal_estimator"),
                        "oneDAL model was not trained.",
                    ),
                ]
            )

            return patching_status

        def _onedal_gpu_supported(self, method_name, *data):
            if method_name == "fit":
                return self._onedal_gpu_fit_supported(method_name, *data)
            if method_name in ["predict", "predict_proba", "predict_log_proba", "score"]:
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

        def _onedal_fit(self, X, y, sample_weight=None, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return self._onedal_cpu_fit(X, y, sample_weight)

            assert sample_weight is None

            if sklearn_check_version("1.0"):
                X, y = self._validate_data(X, y, dtype=[np.float64, np.float32])
            else:
                X, y = check_X_y(X, y, dtype=[np.float64, np.float32])

            self._initialize_onedal_estimator()
            self._onedal_estimator.fit(X, y, queue=queue)
            self._save_attributes()

        def _onedal_predict(self, X, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return daal4py_predict(self, X, "computeClassLabels")

            check_is_fitted(self)
            if sklearn_check_version("1.0"):
                X = self._validate_data(X, reset=False, dtype=[np.float64, np.float32])
            else:
                X = check_array(X, dtype=[np.float64, np.float32])

            assert hasattr(self, "_onedal_estimator")
            return self._onedal_estimator.predict(X, queue=queue)

        def _onedal_predict_proba(self, X, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return daal4py_predict(self, X, "computeClassProbabilities")

            check_is_fitted(self)
            if sklearn_check_version("1.0"):
                X = self._validate_data(X, reset=False, dtype=[np.float64, np.float32])
            else:
                X = check_array(X, dtype=[np.float64, np.float32])

            assert hasattr(self, "_onedal_estimator")
            return self._onedal_estimator.predict_proba(X, queue=queue)

        def _onedal_predict_log_proba(self, X, queue=None):
            if queue is None or queue.sycl_device.is_cpu:
                return daal4py_predict(self, X, "computeClassLogProbabilities")

            check_is_fitted(self)
            if sklearn_check_version("1.0"):
                X = self._validate_data(X, reset=False, dtype=[np.float64, np.float32])
            else:
                X = check_array(X, dtype=[np.float64, np.float32])

            assert hasattr(self, "_onedal_estimator")
            return self._onedal_estimator.predict_log_proba(X, queue=queue)

        fit.__doc__ = sklearn_LogisticRegression.fit.__doc__
        predict.__doc__ = sklearn_LogisticRegression.predict.__doc__
        predict_proba.__doc__ = sklearn_LogisticRegression.predict_proba.__doc__
        predict_log_proba.__doc__ = sklearn_LogisticRegression.predict_log_proba.__doc__
        score.__doc__ = sklearn_LogisticRegression.score.__doc__

else:
    LogisticRegression = LogisticRegression_daal4py

    logging.warning(
        "Sklearnex LogisticRegression requires oneDAL version >= 2024.0.1 "
        "but it was not found"
    )
