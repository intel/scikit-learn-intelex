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

from abc import ABC

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils import _column_or_1d

from .._utils import PatchingConditionsChain


def get_dual_coef(self):
    return self.dual_coef_


def set_dual_coef(self, value):
    self.dual_coef_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.dual_coef_ = value
        if not self._is_in_fit:
            del self._onedal_estimator._onedal_model


def get_intercept(self):
    return self._intercept_


def set_intercept(self, value):
    self._intercept_ = value
    if hasattr(self, "_onedal_estimator"):
        self._onedal_estimator.intercept_ = value
        if not self._is_in_fit:
            del self._onedal_estimator._onedal_model


class BaseSVM(ABC):
    def _onedal_gpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(f"sklearn.{method_name}")
        patching_status.and_conditions([(False, "GPU offloading is not supported.")])
        return patching_status

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        if method_name == "fit":
            patching_status.and_conditions(
                [
                    (
                        self.kernel in ["linear", "rbf", "poly", "sigmoid"],
                        f'Kernel is "{self.kernel}" while '
                        '"linear", "rbf", "poly" and "sigmoid" are only supported.',
                    )
                ]
            )
            return patching_status
        inference_methods = (
            ["predict"]
            if class_name.endswith("R")
            else ["predict", "predict_proba", "decision_function", "score"]
        )
        if method_name in inference_methods:
            patching_status.and_conditions(
                [(hasattr(self, "_onedal_estimator"), "oneDAL model was not trained.")]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {class_name}")


class BaseSVC(BaseSVM):
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
        params = self.get_params()
        params["probability"] = False
        params["decision_function_shape"] = "ovr"
        clf_base = self.__class__(**params)

        try:
            n_splits = 5
            n_jobs = n_splits if queue is None or queue.sycl_device.is_cpu else 1
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            self.clf_prob = CalibratedClassifierCV(
                clf_base, ensemble=False, cv=cv, method="sigmoid", n_jobs=n_jobs
            )
            self.clf_prob.fit(X, y, sample_weight)
        except ValueError:
            clf_base = clf_base.fit(X, y, sample_weight)
            self.clf_prob = CalibratedClassifierCV(
                clf_base, cv="prefit", method="sigmoid"
            )
            self.clf_prob.fit(X, y, sample_weight)

    def _save_attributes(self):
        self.support_vectors_ = self._onedal_estimator.support_vectors_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.shape_fit_ = self._onedal_estimator.class_weight_
        self.classes_ = self._onedal_estimator.classes_
        self.class_weight_ = self._onedal_estimator.class_weight_
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

        if sklearn_check_version("1.1"):
            length = int(len(self.classes_) * (len(self.classes_) - 1) / 2)
            self.n_iter_ = np.full((length,), self._onedal_estimator.n_iter_)


class BaseSVR(BaseSVM):
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

        if sklearn_check_version("1.1"):
            self.n_iter_ = self._onedal_estimator.n_iter_
