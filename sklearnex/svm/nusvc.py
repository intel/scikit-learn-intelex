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

from scipy import sparse as sp
import logging
from .._utils import get_patch_message
from ._common import BaseSVC

from sklearn.svm import NuSVC as sklearn_NuSVC
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.exceptions import NotFittedError

from onedal.svm import NuSVC as onedal_NuSVC


class NuSVC(sklearn_NuSVC, BaseSVC):
    @_deprecate_positional_args
    def __init__(self, *, nu=0.5, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        super().__init__(
            nu=nu, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape, break_ties=break_ties,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        if self.kernel in ['linear', 'rbf', 'poly'] and not sp.isspmatrix(X):
            logging.info("sklearn.svm.NuSVC.fit: " + get_patch_message("onedal"))
            self._onedal_fit(X, y, sample_weight)
        else:
            logging.info("sklearn.svm.NuSVC.fit: " + get_patch_message("sklearn"))
            sklearn_NuSVC.fit(self, X, y, sample_weight)

        return self

    def predict(self, X):
        if hasattr(self, '_onedal_estimator') and not sp.isspmatrix(X):
            logging.info("sklearn.svm.NuSVC.predict: " + get_patch_message("onedal"))
            return self._onedal_estimator.predict(X)
        else:
            logging.info("sklearn.svm.NuSVC.predict: " + get_patch_message("sklearn"))
            return sklearn_NuSVC.predict(self, X)

    def _predict_proba(self, X):
        if hasattr(self, '_onedal_estimator') and not sp.isspmatrix(X):
            logging.info(
                "sklearn.svm.NuSVC._predict_proba: " + get_patch_message("onedal"))
            if getattr(self, 'clf_prob', None) is None:
                raise NotFittedError(
                    "predict_proba is not available when fitted with probability=False")
            return self.clf_prob.predict_proba(X)
        else:
            logging.info(
                "sklearn.svm.NuSVC._predict_proba: " + get_patch_message("sklearn"))
            return sklearn_NuSVC._predict_proba(self, X)

    def decision_function(self, X):
        if hasattr(self, '_onedal_estimator') and not sp.isspmatrix(X):
            logging.info(
                "sklearn.svm.NuSVC.decision_function: " + get_patch_message("onedal"))
            return self._onedal_estimator.decision_function(X)
        else:
            logging.info(
                "sklearn.svm.NuSVC.decision_function: " + get_patch_message("sklearn"))
            return sklearn_NuSVC.decision_function(self, X)

    def _onedal_fit(self, X, y, sample_weight=None):
        onedal_params = {
            'nu': self.nu,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'break_ties': self.break_ties,
            'decision_function_shape': self.decision_function_shape,
        }

        self._onedal_estimator = onedal_NuSVC(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight)

        if self.class_weight == 'balanced':
            self.class_weight_ = self._compute_balanced_class_weight(y)
        else:
            self.class_weight_ = self._onedal_estimator.class_weight_

        if self.probability:
            self._fit_proba(X, y, sample_weight)
        self._save_attributes()
