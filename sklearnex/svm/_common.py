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

from abc import ABC
import numpy as np
try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn import __version__ as sklearn_version

from onedal.datatypes.validation import _column_or_1d


def get_dual_coef(self):
    return self.dual_coef_


def set_dual_coef(self, value):
    self.dual_coef_ = value
    if hasattr(self, '_onedal_estimator'):
        self._onedal_estimator.dual_coef_ = value
        if not self._is_in_fit:
            del self._onedal_estimator._onedal_model


def get_intercept(self):
    return self._intercept_


def set_intercept(self, value):
    self._intercept_ = value
    if hasattr(self, '_onedal_estimator'):
        self._onedal_estimator.intercept_ = value
        if not self._is_in_fit:
            del self._onedal_estimator._onedal_model


class BaseSVC(ABC):
    def _compute_balanced_class_weight(self, y):
        y_ = _column_or_1d(y)
        classes, _ = np.unique(y_, return_inverse=True)

        le = LabelEncoder()
        y_ind = le.fit_transform(y_)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y_) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        return recip_freq[le.transform(classes)]

    def _fit_proba(self, X, y, sample_weight=None, queue=None):
        from .._config import get_config, config_context

        params = self.get_params()
        params["probability"] = False
        params["decision_function_shape"] = 'ovr'
        clf_base = self.__class__(**params)

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg['target_offload'] = queue
        with config_context(**cfg):
            try:
                n_splits = 5
                n_jobs = n_splits if queue is None or queue.sycl_device.is_cpu else 1
                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state)
                if Version(sklearn_version) >= Version("0.24"):
                    self.clf_prob = CalibratedClassifierCV(
                        clf_base, ensemble=False, cv=cv, method='sigmoid',
                        n_jobs=n_jobs)
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


class BaseSVR(ABC):
    def _save_attributes(self):
        self.support_vectors_ = self._onedal_estimator.support_vectors_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.shape_fit_ = self._onedal_estimator.shape_fit_
        self.support_ = self._onedal_estimator.support_

        self._intercept_ = self._onedal_estimator.intercept_
        self._n_support = [self.support_vectors_.shape[0]]
        self._sparse = False
        self._gamma = self._onedal_estimator._gamma
        self._probA = None
        self._probB = None

        self._dual_coef_ = property(get_dual_coef, set_dual_coef)
        self.intercept_ = property(get_intercept, set_intercept)

        self._is_in_fit = True
        self._dual_coef_ = self.dual_coef_
        self.intercept_ = self._intercept_
        self._is_in_fit = False
