# ==============================================================================
# Copyright 2021 Intel Corporation
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
# ==============================================================================

import numpy as np
from scipy import sparse as sp
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as _sklearn_SVC
from sklearn.utils.validation import (
    _deprecate_positional_args,
    check_array,
    check_is_fitted,
)

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version

from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain
from ..utils._array_api import get_namespace
from ._common import BaseSVC

if sklearn_check_version("1.0"):
    from sklearn.utils.metaestimators import available_if

from onedal.svm import SVC as onedal_SVC

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data
else:
    validate_data = _sklearn_SVC._validate_data


@control_n_jobs(
    decorated_methods=["fit", "predict", "_predict_proba", "decision_function", "score"]
)
class SVC(BaseSVC, _sklearn_SVC):
    __doc__ = _sklearn_SVC.__doc__
    _onedal_factory = onedal_SVC

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_SVC._parameter_constraints}

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super().__init__(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        elif self.C <= 0:
            # else if added to correct issues with
            # sklearn tests:
            # svm/tests/test_sparse.py::test_error
            # svm/tests/test_svm.py::test_bad_input
            # for sklearn versions < 1.2 (i.e. without
            # validate_params parameter checking)
            # Without this, a segmentation fault with
            # Windows fatal exception: access violation
            # occurs
            raise ValueError("C <= 0")
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_SVC.fit,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

        return self

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        if len(data) > 1:
            self._class_count = len(np.unique(data[1]))
        self._is_sparse = sp.issparse(data[0])
        conditions = [
            (
                self.kernel in ["linear", "rbf"],
                f'Kernel is "{self.kernel}" while '
                '"linear" and "rbf" are only supported on GPU.',
            ),
            (self.class_weight is None, "Class weight is not supported on GPU."),
            (not self._is_sparse, "Sparse input is not supported on GPU."),
            (self._class_count == 2, "Multiclassification is not supported on GPU."),
        ]
        if method_name == "fit":
            patching_status.and_conditions(conditions)
            return patching_status
        if method_name in ["predict", "predict_proba", "decision_function", "score"]:
            conditions.append(
                (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained")
            )
            patching_status.and_conditions(conditions)
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {class_name}")

    def _get_sample_weight(self, X, y, sample_weight=None):
        sample_weight = super()._get_sample_weight(X, y, sample_weight)
        if sample_weight is None:
            return sample_weight

        if np.any(sample_weight <= 0) and len(np.unique(y[sample_weight > 0])) != len(
            self.classes_
        ):
            raise ValueError(
                "Invalid input - all samples with positive weights "
                "belong to the same class"
                if sklearn_check_version("1.2")
                else "Invalid input - all samples with positive weights "
                "have the same label."
            )
        return sample_weight

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        X, _, weights = self._onedal_fit_checks(X, y, sample_weight)
        onedal_params = {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self._compute_gamma_sigma(X),
            "coef0": self.coef0,
            "tol": self.tol,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "break_ties": self.break_ties,
            "decision_function_shape": self.decision_function_shape,
        }

        self._onedal_estimator = self._onedal_factory(**onedal_params)
        self._onedal_estimator.fit(X, y, weights, queue=queue)

        if self.probability:
            self._fit_proba(
                X,
                y,
                sample_weight=sample_weight,
                queue=queue,
            )

        self._save_attributes()

    fit.__doc__ = _sklearn_SVC.fit.__doc__
