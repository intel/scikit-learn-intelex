#===============================================================================
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
#===============================================================================

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from abc import ABCMeta, abstractmethod
from enum import Enum
import sys

import numpy as np
from scipy import sparse as sp
from ..common import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _get_sample_weight,
    _check_is_fitted,
    _column_or_1d,
    _check_n_features
)

try:
    from _onedal4py_dpc import (
        PySvmParams,
        PyRegressionSvmTrain,
        PyRegressionSvmInfer,
        PyClassificationSvmTrain,
        PyClassificationSvmInfer
    )
except ImportError:
    from _onedal4py_host import (
        PySvmParams,
        PyRegressionSvmTrain,
        PyRegressionSvmInfer,
        PyClassificationSvmTrain,
        PyClassificationSvmInfer
    )


class SVMtype(Enum):
    c_svc = 0
    epsilon_svr = 1


class BaseSVM(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, C, epsilon, kernel='rbf', *, degree, gamma,
                 coef0, tol, shrinking, cache_size, max_iter, tau,
                 class_weight, decision_function_shape,
                 break_ties, algorithm, svm_type=None, **kwargs):

        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.tol = tol
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.tau = tau
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.algorithm = algorithm
        self.svm_type = svm_type

    def _compute_gamma_sigma(self, gamma, X):
        if isinstance(gamma, str):
            if gamma == 'scale':
                if sp.isspmatrix(X):
                    # var = E[X^2] - E[X]^2
                    X_sc = (X.multiply(X)).mean() - (X.mean())**2
                else:
                    X_sc = X.var()
                _gamma = 1.0 / (X.shape[1] * X_sc) if X_sc != 0 else 1.0
            elif gamma == 'auto':
                _gamma = 1.0 / X.shape[1]
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(gamma)
                )
        else:
            _gamma = gamma
        return _gamma, np.sqrt(0.5 / _gamma)

    def _validate_targets(self, y, dtype):
        self.class_weight_ = None
        self.classes_ = None
        return _column_or_1d(y, warn=True).astype(dtype, copy=False)

    def _get_onedal_params(self):
        max_iter = 10000 if self.max_iter == -1 else self.max_iter
        class_count = 0 if self.classes_ is None else len(self.classes_)
        return PySvmParams(method=self.algorithm, kernel=self.kernel,
                           c=self.C, epsilon=self.epsilon,
                           class_count=class_count, accuracy_threshold=self.tol,
                           max_iteration_count=max_iter,
                           scale=self._scale_, sigma=self._sigma_,
                           shift=self.coef0, degree=self.degree, tau=self.tau)

    def _reset_context(func):
        def wrapper(*args, **kwargs):
            if 'daal4py.oneapi' in sys.modules:
                import daal4py.oneapi as d4p_oneapi
                devname = d4p_oneapi._get_device_name_sycl_ctxt()
                ctxparams = d4p_oneapi._get_sycl_ctxt_params()

                if devname == 'gpu' and ctxparams.get('host_offload_on_fail', False):
                    gpu_ctx = d4p_oneapi._get_sycl_ctxt()
                    host_ctx = d4p_oneapi.sycl_execution_context('host')
                    try:
                        host_ctx.apply()
                        res = func(*args, **kwargs)
                    finally:
                        del host_ctx
                        gpu_ctx.apply()
                    return res
                else:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper

    @_reset_context
    def _fit(self, X, y, sample_weight, Computer):
        if hasattr(self, 'decision_function_shape'):
            if self.decision_function_shape not in ('ovr', 'ovo', None):
                raise ValueError(
                    f"decision_function_shape must be either 'ovr' or 'ovo', "
                    f"got {self.decision_function_shape}."
                )

        if y is None:
            if self._get_tags()['requires_y']:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
        X, y = _check_X_y(
            X, y, dtype=[np.float64, np.float32],
            force_all_finite=True, accept_sparse='csr')
        y = self._validate_targets(y, X.dtype)
        sample_weight = _get_sample_weight(
            X, y, sample_weight, self.class_weight_, self.classes_, self.svm_type)

        self._sparse = sp.isspmatrix(X)

        if self.kernel == 'linear':
            self._scale_, self._sigma_ = 1.0, 1.0
            self.coef0 = 0.0
        else:
            self._scale_, self._sigma_ = self._compute_gamma_sigma(self.gamma, X)

        c_svm = Computer(self._get_onedal_params())
        c_svm.train(X, y, sample_weight)

        if self._sparse:
            self.dual_coef_ = sp.csr_matrix(c_svm.get_coeffs().T)
            self.support_vectors_ = sp.csr_matrix(c_svm.get_support_vectors())
        else:
            self.dual_coef_ = c_svm.get_coeffs().T
            self.support_vectors_ = c_svm.get_support_vectors()

        self.intercept_ = c_svm.get_biases().ravel()
        self.support_ = c_svm.get_support_indices().ravel().astype('int')
        self.n_features_in_ = X.shape[1]
        self.shape_fit_ = X.shape

        if getattr(self, 'classes_', None) is not None:
            self._n_support = np.array([
                np.sum(y[self.support_] == label) for label in self.classes_])
        self._gamma = self._scale_

        self._onedal_model = c_svm.get_model()
        return self

    @_reset_context
    def _predict(self, X, Computer):
        _check_is_fitted(self)
        if self.break_ties and self.decision_function_shape == 'ovo':
            raise ValueError("break_ties must be False when "
                             "decision_function_shape is 'ovo'")

        if self.break_ties and self.decision_function_shape == 'ovr' and \
                len(self.classes_) > 2:
            y = np.argmax(self.decision_function(X), axis=1)
        else:
            X = _check_array(X, dtype=[np.float64, np.float32],
                             force_all_finite=True, accept_sparse='csr')
            _check_n_features(self, X, False)

            if self._sparse and not sp.isspmatrix(X):
                X = sp.csr_matrix(X)
            if self._sparse:
                X.sort_indices()

            if sp.issparse(X) and not self._sparse and not callable(self.kernel):
                raise ValueError(
                    "cannot use sparse input in %r trained on dense data"
                    % type(self).__name__)

            c_svm = Computer(self._get_onedal_params())

            if hasattr(self, '_onedal_model'):
                c_svm.infer(X, self._onedal_model)
            else:
                c_svm.infer_builder(X, self.support_vectors_,
                                    self.dual_coef_.T, self.intercept_)
            y = c_svm.get_labels()
        return y

    def _ovr_decision_function(self, predictions, confidences, n_classes):
        n_samples = predictions.shape[0]
        votes = np.zeros((n_samples, n_classes))
        sum_of_confidences = np.zeros((n_samples, n_classes))

        k = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                sum_of_confidences[:, i] -= confidences[:, k]
                sum_of_confidences[:, j] += confidences[:, k]
                votes[predictions[:, k] == 0, i] += 1
                votes[predictions[:, k] == 1, j] += 1
                k += 1

        transformed_confidences = \
            sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1))
        return votes + transformed_confidences

    @_reset_context
    def _decision_function(self, X):
        _check_is_fitted(self)
        X = _check_array(X, dtype=[np.float64, np.float32],
                         force_all_finite=False, accept_sparse='csr')
        _check_n_features(self, X, False)

        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)

        c_svm = PyClassificationSvmInfer(self._get_onedal_params())
        if hasattr(self, '_onedal_model'):
            c_svm.infer(X, self._onedal_model)
        else:
            c_svm.infer_builder(X, self.support_vectors_,
                                self.dual_coef_.T, self.intercept_)
        decision_function = c_svm.get_decision_function()
        if len(self.classes_) == 2:
            decision_function = decision_function.ravel()

        if self.decision_function_shape == 'ovr' and len(self.classes_) > 2:
            decision_function = self._ovr_decision_function(
                decision_function < 0, -decision_function, len(self.classes_))
        return decision_function


class SVR(RegressorMixin, BaseSVM):
    """
    Epsilon--Support Vector Regression.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', *, degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, shrinking=True,
                 cache_size=200.0, max_iter=-1, tau=1e-12,
                 algorithm='thunder', **kwargs):
        super().__init__(C=C, epsilon=epsilon, kernel=kernel,
                         degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol,
                         shrinking=shrinking, cache_size=cache_size,
                         max_iter=max_iter, tau=tau, class_weight=None,
                         decision_function_shape=None,
                         break_ties=False, algorithm=algorithm)
        self.svm_type = SVMtype.epsilon_svr

    def fit(self, X, y, sample_weight=None):
        return super()._fit(X, y, sample_weight, PyRegressionSvmTrain)

    def predict(self, X):
        y = super()._predict(X, PyRegressionSvmInfer)
        return y.ravel()


class SVC(ClassifierMixin, BaseSVM):
    """
    C-Support Vector Classification.
    """

    def __init__(self, C=1.0, kernel='rbf', *, degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, shrinking=True, cache_size=200.0,
                 max_iter=-1, tau=1e-12, class_weight=None,
                 decision_function_shape='ovr', break_ties=False,
                 algorithm='thunder', **kwargs):
        super().__init__(C=C, epsilon=0.0, kernel=kernel, degree=degree,
                         gamma=gamma, coef0=coef0, tol=tol,
                         shrinking=shrinking, cache_size=cache_size,
                         max_iter=max_iter, tau=tau, class_weight=class_weight,
                         decision_function_shape=decision_function_shape,
                         break_ties=break_ties, algorithm=algorithm)
        self.svm_type = SVMtype.c_svc

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype)
        return y

    def fit(self, X, y, sample_weight=None):
        return super()._fit(X, y, sample_weight, PyClassificationSvmTrain)

    def predict(self, X):
        y = super()._predict(X, PyClassificationSvmInfer)
        if len(self.classes_) == 2:
            y = y.ravel()
        return self.classes_.take(np.asarray(y, dtype=np.intp)).ravel()

    def decision_function(self, X):
        return super()._decision_function(X)
