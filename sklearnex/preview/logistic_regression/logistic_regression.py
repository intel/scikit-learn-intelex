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

from ..._device_offload import dispatch, wrap_output_data
from ...utils.validation import _assert_all_finite

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
import logging
from scipy import sparse as sp
import numpy as np
from daal4py.oneapi import sycl_context
from daal4py.sklearn._utils import (
    get_dtype, make2d, PatchingConditionsChain)

from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression


class LogisticRegression(sklearn_LogisticRegression):
    __doc__ = sklearn_LogisticRegression.__doc__
    __parameter_constraints = dict = {**sklearn_LogisticRegression._parameter_constraints}

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
        solver=("lbfgs" if sklearn_check_version('0.22') else 'liblinear'),
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None
    ):
        super().__init__(penalty=penalty,
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
                         l1_ratio=l1_ratio)

    def fit_onedal(self, X, y, sample_weight=None, queue=None):
        return super().fit(X, y, sample_weight)

    def fit(self, X, y, sample_weight=None):
        dispatch(self, 'fit', {
            'onedal': self.__class__.fit_onedal,
            'sklearn': sklearn_LogisticRegression.fit,
        }, X, y, sample_weight)
        return self

    def predict_onedal(self, X, queue=None):
        return super().predict(X)

    @wrap_output_data
    def predict(self, X, queue=None):
        result = dispatch(self, 'predict', {
            'onedal': self.__class__.predict_onedal,
            'sklearn': sklearn_LogisticRegression.predict,
        }, X)
        return result

    def predict_proba_onedal(self, X, queue=None):
        return super().predict_proba(X)

    @wrap_output_data
    def predict_proba(self, X):
        result = dispatch(self, 'predict_proba', {
            'onedal': self.__class__.predict_proba_onedal,
            'sklearn': sklearn_LogisticRegression.predict_proba,
        }, X)
        return result

    def predict_log_proba_onedal(self, X, queue=None):
        return super().predict_log_proba(X)

    @wrap_output_data
    def predict_log_proba(self, X):
        result = dispatch(self, 'predict_log_proba', {
            'onedal': self.__class__.predict_log_proba_onedal,
            'sklearn': sklearn_LogisticRegression.predict_log_proba,
        }, X)
        return result

    def decision_function_onedal(self, X, queue=None):
        return super().decision_function(X)

    @wrap_output_data
    def decision_function(self, X):
        result = dispatch(self, 'decision_function', {
            'onedal': self.__class__.decision_function_onedal,
            'sklearn': sklearn_LogisticRegression.decision_function,
        }, X)
        return result

    def score_onedal(X, y, sample_weight=None, queue=None):
        return super().score(X, y, sample_weight)

    def score(self, X, y, sample_weight=None):
        result = dispatch(self, 'fit', {
            'onedal': self.__class__.score_onedal,
            'sklearn': sklearn_LogisticRegression.score,
        }, X, y, sample_weight)
        return result

    def _onedal_supported(self, method_name, *data):
        return True

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)
