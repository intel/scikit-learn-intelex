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

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse as sp

from onedal import _backend

from ..common._base import BaseEstimator
from ..common._estimator_checks import _check_is_fitted
from ..common._mixin import ClassifierMixin, RegressorMixin
from ..common._policy import _get_policy
from ..datatypes import _convert_to_supported, from_table, to_table
from ..utils import (
    _check_array,
    _check_n_features,
    _check_X_y,
    _column_or_1d,
    _validate_targets,
)


class BaseSVM(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        C,
        nu,
        epsilon,
        kernel="rbf",
        *,
        degree,
        gamma,
        coef0,
        tol,
        shrinking,
        cache_size,
        max_iter,
        tau,
        class_weight,
        decision_function_shape,
        break_ties,
        algorithm,
    ):
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

    def _validate_targets(self, y, dtype):
        self.class_weight_ = None
        self.classes_ = None
        return _column_or_1d(y, warn=True).astype(dtype, copy=False)

    def _get_onedal_params(self, dtype):
        max_iter = 10000 if self.max_iter == -1 else self.max_iter
        # TODO: remove this workaround
        # when oneDAL SVM starts support of 'n_iterations' result
        self.n_iter_ = 1 if max_iter < 1 else max_iter
        class_count = 0 if self.classes_ is None else len(self.classes_)
        return {
            "fptype": dtype,
            "method": self.algorithm,
            "kernel": self.kernel,
            "c": self.C,
            "nu": self.nu,
            "epsilon": self.epsilon,
            "class_count": class_count,
            "accuracy_threshold": self.tol,
            "max_iteration_count": int(max_iter),
            "scale": self._scale_,
            "sigma": self._sigma_,
            "shift": self.coef0,
            "degree": self.degree,
            "tau": self.tau,
            "shrinking": self.shrinking,
            "cache_size": self.cache_size,
        }

    def _fit(self, X, y, sample_weight, module, queue):
        if hasattr(self, "decision_function_shape"):
            if self.decision_function_shape not in ("ovr", "ovo", None):
                raise ValueError(
                    f"decision_function_shape must be either 'ovr' or 'ovo', "
                    f"got {self.decision_function_shape}."
                )

        X, y = _check_X_y(
            X,
            y,
            dtype=[np.float64, np.float32],
            force_all_finite=True,
            accept_sparse="csr",
        )
        # hard work remains on moving validate targets away from onedal
        y = self._validate_targets(y, X.dtype)
        if sample_weight is not None and len(sample_weight) > 0:
            sample_weight = _check_array(
                sample_weight,
                accept_sparse=False,
                ensure_2d=False,
                dtype=X.dtype,
                order="C",
            )
        elif self.class_weight is not None:
            sample_weight = np.ones(X.shape[0], dtype=X.dtype)

        if sample_weight is not None:
            if self.class_weight_ is not None:
                for i, v in enumerate(self.class_weight_):
                    sample_weight[y == i] *= v
            data = (X, y, sample_weight)
        else:
            data = (X, y)
        self._sparse = sp.issparse(X)

        if self.kernel == "linear":
            self._scale_, self._sigma_ = 1.0, 1.0
            self.coef0 = 0.0
        else:
            self._scale_, self._sigma_ = self.gamma, np.sqrt(0.5 / self.gamma)

        policy = _get_policy(queue, *data)
        data_t = to_table(*_convert_to_supported(policy, *data))
        params = self._get_onedal_params(data_t[0].dtype)
        result = module.train(policy, params, *data_t)

        if self._sparse:
            self.dual_coef_ = sp.csr_matrix(from_table(result.coeffs).T)
            self.support_vectors_ = sp.csr_matrix(from_table(result.support_vectors))
        else:
            self.dual_coef_ = from_table(result.coeffs).T
            self.support_vectors_ = from_table(result.support_vectors)

        self.intercept_ = from_table(result.biases).ravel()
        self.support_ = from_table(result.support_indices).ravel().astype("int")
        self.n_features_in_ = X.shape[1]
        self.shape_fit_ = X.shape

        # _n_support not used in this object, will be moved to sklearnex
        if getattr(self, "classes_", None) is not None:
            indices = y.take(self.support_, axis=0)
            self._n_support = np.array(
                [np.sum(indices == i) for i, _ in enumerate(self.classes_)]
            )
        self._gamma = self._scale_

        self._onedal_model = result.model
        return self

    def _create_model(self, module):
        m = module.model()

        m.support_vectors = to_table(self.support_vectors_)
        m.coeffs = to_table(self.dual_coef_.T)
        m.biases = to_table(self.intercept_)
        return m

    def _infer(self, X, module, queue):
        _check_is_fitted(self)

        if self._sparse:
            if not sp.isspmatrix(X):
                X = sp.csr_matrix(X)
            else:
                X.sort_indices()
        elif sp.issparse(X) and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__
            )

        policy = _get_policy(queue, X)
        X = to_table(_convert_to_supported(policy, X))
        params = self._get_onedal_params(X.dtype)

        if hasattr(self, "_onedal_model"):
            model = self._onedal_model
        else:
            model = self._create_model(module)
        return module.infer(policy, params, model, X)

    def _predict(self, X, module, queue):
        return from_table(self._infer(X, module, queue).responses)

    def _decision_function(self, X, module, queue):
        return from_table(self._infer(X, module, queue).decision_function)


class SVR(RegressorMixin, BaseSVM):
    """
    Epsilon--Support Vector Regression.
    """

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=0.5,
            epsilon=epsilon,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=None,
            decision_function_shape=None,
            break_ties=False,
            algorithm=algorithm,
        )

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.regression, queue)

    def predict(self, X, queue=None):
        return super()._predict(X, _backend.svm.regression, queue)


class SVC(ClassifierMixin, BaseSVM):
    """
    C-Support Vector Classification.
    """

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        class_weight=None,
        decision_function_shape="ovr",
        break_ties=False,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=0.5,
            epsilon=0.0,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=class_weight,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            algorithm=algorithm,
        )

    def _create_model(self, module):
        m = super()._create_model(module)
        m.first_class_response, m.second_class_response = 0, 1
        return m

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype
        )
        return y

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.classification, queue)

    def predict(self, X, queue=None):
        return super()._predict(X, _backend.svm.classification, queue)

    def decision_function(self, X, queue=None):
        return super()._decision_function(X, _backend.svm.classification, queue)


class NuSVR(RegressorMixin, BaseSVM):
    """
    Nu-Support Vector Regression.
    """

    def __init__(
        self,
        nu=0.5,
        C=1.0,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        algorithm="thunder",
    ):
        super().__init__(
            C=C,
            nu=nu,
            epsilon=0.0,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=None,
            decision_function_shape=None,
            break_ties=False,
            algorithm=algorithm,
        )

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.nu_regression, queue)

    def predict(self, X, queue=None):
        return super()._predict(X, _backend.svm.nu_regression, queue)


class NuSVC(ClassifierMixin, BaseSVM):
    """
    Nu-Support Vector Classification.
    """

    def __init__(
        self,
        nu=0.5,
        kernel="rbf",
        *,
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        shrinking=True,
        cache_size=200.0,
        max_iter=-1,
        tau=1e-12,
        class_weight=None,
        decision_function_shape="ovr",
        break_ties=False,
        algorithm="thunder",
    ):
        super().__init__(
            C=1.0,
            nu=nu,
            epsilon=0.0,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            shrinking=shrinking,
            cache_size=cache_size,
            max_iter=max_iter,
            tau=tau,
            class_weight=class_weight,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            algorithm=algorithm,
        )

    def _create_model(self, module):
        m = super()._create_model(module)
        m.first_class_response, m.second_class_response = 0, 1
        return m

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype
        )
        return y

    def fit(self, X, y, sample_weight=None, queue=None):
        return super()._fit(X, y, sample_weight, _backend.svm.nu_classification, queue)

    def predict(self, X, queue=None):
        return super()._predict(X, _backend.svm.nu_classification, queue)

    def decision_function(self, X, queue=None):
        return super()._decision_function(X, _backend.svm.nu_classification, queue)
