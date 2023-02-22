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

from abc import ABC
import numpy as np
from daal4py.sklearn._utils import sklearn_check_version

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from onedal.datatypes.validation import _column_or_1d


def get_coef(self):
    return self._coef_


def set_coef(self, value):
    self._coef_ = value
    if hasattr(self, '_onedal_estimator'):
        self._onedal_estimator.coef_ = value
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


class BaseLinearRegression(ABC):
    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self._coef_ = self._onedal_estimator.coef_
        self._intercept_ = self._onedal_estimator.intercept_
        self._sparse = False

        self.coef_ = property(get_coef, set_coef)
        self.intercept_ = property(get_intercept, set_intercept)

        self._is_in_fit = True
        self.coef_ = self._coef_
        self.intercept_ = self._intercept_
        self._is_in_fit = False
