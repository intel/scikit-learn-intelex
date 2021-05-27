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

import numpy as np
import sys
from scipy import sparse as sp
import logging
from distutils.version import LooseVersion
from .._utils import get_patch_message
from ._common import get_dual_coef, set_dual_coef, get_intercept, set_intercept

from sklearn.svm import SVC as sklearn_SVC
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn import __version__ as sklearn_version

from onedal.svm import SVC as onedal_SVC
from onedal.common.validation import _column_or_1d


class SVC(sklearn_SVC):
    @_deprecate_positional_args
    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        super().__init__(
            C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape, break_ties=break_ties,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        if self.kernel in ['linear', 'rbf', 'poly']:
            logging.info("sklearn.svm.SVC.fit: " + get_patch_message("onedal"))
            self._onedal_fit(X, y, sample_weight)
        else:
            logging.info("sklearn.svm.SVC.fit: " + get_patch_message("sklearn"))
            sklearn_SVC.fit(self, X, y, sample_weight)

        return self

    def predict(self, X):
        if hasattr(self, '_onedal_estimator'):
            logging.info("sklearn.svm.SVC.predict: " + get_patch_message("onedal"))
            return self._onedal_estimator.predict(X)
        else:
            logging.info("sklearn.svm.SVC.predict: " + get_patch_message("sklearn"))
            return sklearn_SVC.predict(self, X)

    def _predict_proba(self, X):
        if hasattr(self, '_onedal_estimator'):
            logging.info("sklearn.svm.SVC._predict_proba: " + get_patch_message("onedal"))
            if getattr(self, 'clf_prob', None) is None:
                raise NotFittedError(
                    "predict_proba is not available when fitted with probability=False")
            return self.clf_prob.predict_proba(X)
        else:
            logging.info(
                "sklearn.svm.SVC._predict_proba: " + get_patch_message("sklearn"))
            return sklearn_SVC._predict_proba(self, X)

    def decision_function(self, X):
        if hasattr(self, '_onedal_estimator'):
            logging.info(
                "sklearn.svm.SVC.decision_function: " + get_patch_message("onedal"))
            return self._onedal_estimator.decision_function(X)
        else:
            logging.info(
                "sklearn.svm.SVC.decision_function: " + get_patch_message("sklearn"))
            return sklearn_SVC.decision_function(self, X)

    def _onedal_fit(self, X, y, sample_weight=None):
        onedal_params = {
            'C': self.C,
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

        self._onedal_estimator = onedal_SVC(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight)

        if self.class_weight == 'balanced':
            self.class_weight_ = self._compute_balanced_class_weight(y)
        else:
            self.class_weight_ = self._onedal_estimator.class_weight_

        if self.probability:
            self._fit_proba(X, y, sample_weight)
        self._save_attributes()

    def _compute_balanced_class_weight(self, y):
        y_ = _column_or_1d(y)
        classes, _ = np.unique(y_, return_inverse=True)

        le = LabelEncoder()
        y_ind = le.fit_transform(y_)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y_) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        return recip_freq[le.transform(classes)]

    def _fit_proba(self, X, y, sample_weight=None):
        params = self.get_params()
        params["probability"] = False
        params["decision_function_shape"] = 'ovr'
        clf_base = SVC(**params)
        try:
            n_splits = 5
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state)
            if LooseVersion(sklearn_version) >= LooseVersion("0.24"):
                self.clf_prob = CalibratedClassifierCV(
                    clf_base, ensemble=False, cv=cv, method='sigmoid',
                    n_jobs=n_splits)
            else:
                self.clf_prob = CalibratedClassifierCV(
                    clf_base, cv=cv, method='sigmoid')
            self.clf_prob.fit(X, y, sample_weight)
        except ValueError:
            clf_base = clf_base.fit(X, y, sample_weight)
            self.clf_prob = CalibratedClassifierCV(
                clf_base, cv="prefit", method='sigmoid')
            self.clf_prob.fit(X, y, sample_weight)

    def _save_attributes(self):
        self.support_vectors_ = self._onedal_estimator.support_vectors_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.shape_fit_ = self._onedal_estimator.class_weight_
        self.classes_ = self._onedal_estimator.classes_
        self.support_ = self._onedal_estimator.support_

        self._intercept_ = self._onedal_estimator.intercept_
        self._n_support = self._onedal_estimator._n_support
        self._sparse = False
        self._gamma = self._onedal_estimator._gamma
        if self.probability:
            length = int(len(self.classes_) * (len(self.classes_) - 1) / 2)
            self._probA = np.zeros(length)
            self._probB = np.zeros(length)
        else:
            self._probA = np.empty(0)
            self._probB = np.empty(0)

        self._dual_coef_ = property(get_dual_coef, set_dual_coef)
        self.intercept_ = property(get_intercept, set_intercept)

        self._is_in_fit = True
        self._dual_coef_ = self.dual_coef_
        self.intercept_ = self._intercept_
        self._is_in_fit = False
