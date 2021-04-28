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
