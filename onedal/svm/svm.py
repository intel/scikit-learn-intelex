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

from daal4py.sklearn._utils import sklearn_check_version
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod
from enum import Enum
from numbers import Number, Real

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _column_or_1d,
    _check_n_features
)

from ..common._mixin import ClassifierMixin, RegressorMixin
from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted
from ..datatypes._data_conversion import from_table, to_table
from onedal import _backend


class SVMtype(Enum):
    c_svc = 0
    epsilon_svr = 1
    nu_svc = 2
    nu_svr = 3


class BaseSVM(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, C, nu, epsilon, kernel='rbf', *, degree, gamma,
                 coef0, tol, shrinking, cache_size, max_iter, tau,
                 class_weight, decision_function_shape,
                 break_ties, algorithm, svm_type=None, **kwargs):

        self.C = C
        self.nu = nu
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
            if sklearn_check_version('1.1') and not sklearn_check_version('1.2'):
                if isinstance(gamma, Real):
                    if gamma <= 0:
                        msg = (
                            f"gamma value must be > 0; {gamma!r} is invalid. Use"
                            " a positive number or use 'auto' to set gamma to a"
                            " value of 1 / n_features."
                        )
                        raise ValueError(msg)
                    _gamma = gamma
                else:
                    msg = (
                        "The gamma value should be set to 'scale', 'auto' or a"
                        f" positive float value. {gamma!r} is not a valid option"
                    )
                    raise ValueError(msg)
            else:
                _gamma = gamma
        return _gamma, np.sqrt(0.5 / _gamma)

    def _validate_targets(self, y, dtype):
        self.class_weight_ = None
        self.classes_ = None
        return _column_or_1d(y, warn=True).astype(dtype, copy=False)

    def _get_sample_weight(self, X, y, sample_weight):
        n_samples = X.shape[0]
        dtype = X.dtype
        if n_samples == 1:
            raise ValueError("n_samples=1")

        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)

        sample_weight_count = sample_weight.shape[0]
        if sample_weight_count != 0 and sample_weight_count != n_samples:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                             % (len(sample_weight), X.shape))

        ww = None
        if sample_weight_count == 0 and self.class_weight_ is None:
            return ww

        if sample_weight_count == 0:
            sample_weight = np.ones(n_samples, dtype=dtype)
        elif isinstance(sample_weight, Number):
            sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
        else:
            sample_weight = _check_array(
                sample_weight, accept_sparse=False, ensure_2d=False,
                dtype=dtype, order="C"
            )
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight.shape == {}, expected {}!"
                                 .format(sample_weight.shape, (n_samples,)))

        if self.svm_type == SVMtype.nu_svc:
            weight_per_class = [np.sum(sample_weight[y == class_label])
                                for class_label in np.unique(y)]

            for i in range(len(weight_per_class)):
                for j in range(i + 1, len(weight_per_class)):
                    if self.nu * (weight_per_class[i] + weight_per_class[j]) / 2 > \
                            min(weight_per_class[i], weight_per_class[j]):
                        raise ValueError('specified nu is infeasible')

        if np.all(sample_weight <= 0):
            if self.svm_type == SVMtype.nu_svc:
                err_msg = 'negative dimensions are not allowed'
            else:
                err_msg = 'Invalid input - all samples have zero or negative weights.'
            raise ValueError(err_msg)
        if np.any(sample_weight <= 0):
            if self.svm_type == SVMtype.c_svc and \
                    len(np.unique(y[sample_weight > 0])) != len(self.classes_):
                raise ValueError(
                    'Invalid input - all samples with positive weights '
                    'belong to the same class' if sklearn_check_version('1.2') else
                    'Invalid input - all samples with positive weights '
                    'have the same label.')
        ww = sample_weight
        if self.class_weight_ is not None:
            for i, v in enumerate(self.class_weight_):
                ww[y == i] *= v

        if not ww.flags.c_contiguous and not ww.flags.f_contiguous:
            ww = np.ascontiguousarray(ww, dtype)

        return ww

    def _get_onedal_params(self, data):
        max_iter = 10000 if self.max_iter == -1 else self.max_iter
        # TODO: remove this workaround
        # when oneDAL SVM starts support of 'n_iterations' result
        self.n_iter_ = 1 if max_iter < 1 else max_iter
        class_count = 0 if self.classes_ is None else len(self.classes_)
        return {
            'fptype': 'float' if data.dtype == np.float32 else 'double',
            'method': self.algorithm,
            'kernel': self.kernel,
            'c': self.C, 'nu': self.nu, 'epsilon': self.epsilon,
            'class_count': class_count, 'accuracy_threshold': self.tol,
            'max_iteration_count': int(max_iter), 'scale': self._scale_,
            'sigma': self._sigma_, 'shift': self.coef0, 'degree': self.degree,
            'tau': self.tau, 'shrinking': self.shrinking, 'cache_size': self.cache_size
        }

    def _fit(self, X, y, sample_weight, module, queue):
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
        sample_weight = self._get_sample_weight(X, y, sample_weight)

        self._sparse = sp.isspmatrix(X)

        if self.kernel == 'linear':
            self._scale_, self._sigma_ = 1.0, 1.0
            self.coef0 = 0.0
        else:
            self._scale_, self._sigma_ = self._compute_gamma_sigma(self.gamma, X)

        policy = _get_policy(queue, X, y, sample_weight)
        params = self._get_onedal_params(X)
        result = module.train(policy, params, *to_table(X, y, sample_weight))

        if self._sparse:
            self.dual_coef_ = sp.csr_matrix(from_table(result.coeffs).T)
            self.support_vectors_ = sp.csr_matrix(from_table(result.support_vectors))
        else:
            self.dual_coef_ = from_table(result.coeffs).T
            self.support_vectors_ = from_table(result.support_vectors)

        self.intercept_ = from_table(result.biases).ravel()
        self.support_ = from_table(result.support_indices).ravel().astype('int')
        self.n_features_in_ = X.shape[1]
        self.shape_fit_ = X.shape

        if getattr(self, 'classes_', None) is not None:
            indices = y.take(self.support_, axis=0)
            self._n_support = np.array([
                np.sum(indices == i) for i, _ in enumerate(self.classes_)])
        self._gamma = self._scale_

        self._onedal_model = result.model
        return self

    def _create_model(self, module):
        m = module.model()

        m.support_vectors = to_table(self.support_vectors_)
        m.coeffs = to_table(self.dual_coef_.T)
        m.biases = to_table(self.intercept_)

        if self.svm_type is SVMtype.c_svc or self.svm_type is SVMtype.nu_svc:
            m.first_class_response, m.second_class_response = 0, 1
        return m

    def _predict(self, X, module, queue):
        _check_is_fitted(self)
        if self.break_ties and self.decision_function_shape == 'ovo':
            raise ValueError("break_ties must be False when "
                             "decision_function_shape is 'ovo'")

        if (module in [_backend.svm.classification, _backend.svm.nu_classification]):
            sv = self.support_vectors_
            if not self._sparse and sv.size > 0 and self._n_support.sum() != sv.shape[0]:
                raise ValueError("The internal representation "
                                 f"of {self.__class__.__name__} was altered")

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

            policy = _get_policy(queue, X)
            params = self._get_onedal_params(X)

            if hasattr(self, '_onedal_model'):
                model = self._onedal_model
            else:
                model = self._create_model(module)
            result = module.infer(policy, params, model, to_table(X))
            y = from_table(result.responses)
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

    def _decision_function(self, X, module, queue):
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

        if (module in [_backend.svm.classification, _backend.svm.nu_classification]):
            sv = self.support_vectors_
            if not self._sparse and sv.size > 0 and self._n_support.sum() != sv.shape[0]:
                raise ValueError("The internal representation "
                                 f"of {self.__class__.__name__} was altered")

        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)

        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(module)
        result = module.infer(policy, params, model, to_table(X))
        decision_function = from_table(result.decision_function)

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
        super().__init__(C=C, nu=0.5, epsilon=epsilon, kernel=kernel,
                         degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol,
                         shrinking=shrinking, cache_size=cache_size,
                         max_iter=max_iter, tau=tau, class_weight=None,
                         decision_function_shape=None,
                         break_ties=False, algorithm=algorithm)
        self.svm_type = SVMtype.epsilon_svr

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.regression, queue)

    def predict(self, X, queue=None):
        y = super()._predict(X, _backend.svm.regression, queue)
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
        super().__init__(C=C, nu=0.5, epsilon=0.0, kernel=kernel, degree=degree,
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

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.classification, queue)

    def predict(self, X, queue=None):
        y = super()._predict(X, _backend.svm.classification, queue)
        if len(self.classes_) == 2:
            y = y.ravel()
        return self.classes_.take(np.asarray(y, dtype=np.intp)).ravel()

    def decision_function(self, X, queue=None):
        return super()._decision_function(X, _backend.svm.classification, queue)


class NuSVR(RegressorMixin, BaseSVM):
    """
    Nu-Support Vector Regression.
    """

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', *, degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, shrinking=True,
                 cache_size=200.0, max_iter=-1, tau=1e-12,
                 algorithm='thunder', **kwargs):
        super().__init__(C=C, nu=nu, epsilon=0.0, kernel=kernel,
                         degree=degree, gamma=gamma,
                         coef0=coef0, tol=tol,
                         shrinking=shrinking, cache_size=cache_size,
                         max_iter=max_iter, tau=tau, class_weight=None,
                         decision_function_shape=None,
                         break_ties=False, algorithm=algorithm)
        self.svm_type = SVMtype.nu_svr

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.nu_regression, queue)

    def predict(self, X, queue=None):
        y = super()._predict(X, _backend.svm.nu_regression, queue)
        return y.ravel()


class NuSVC(ClassifierMixin, BaseSVM):
    """
    Nu-Support Vector Classification.
    """

    def __init__(self, nu=0.5, kernel='rbf', *, degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, shrinking=True, cache_size=200.0,
                 max_iter=-1, tau=1e-12, class_weight=None,
                 decision_function_shape='ovr', break_ties=False,
                 algorithm='thunder', **kwargs):
        super().__init__(C=1.0, nu=nu, epsilon=0.0, kernel=kernel, degree=degree,
                         gamma=gamma, coef0=coef0, tol=tol,
                         shrinking=shrinking, cache_size=cache_size,
                         max_iter=max_iter, tau=tau, class_weight=class_weight,
                         decision_function_shape=decision_function_shape,
                         break_ties=break_ties, algorithm=algorithm)
        self.svm_type = SVMtype.nu_svc

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype)
        return y

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.nu_classification, queue)

    def predict(self, X, queue=None):
        y = super()._predict(X, _backend.svm.nu_classification, queue)
        if len(self.classes_) == 2:
            y = y.ravel()
        return self.classes_.take(np.asarray(y, dtype=np.intp)).ravel()

    def decision_function(self, X, queue=None):
        return super()._decision_function(X, _backend.svm.nu_classification, queue)
