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

import logging
from .._utils import get_patch_message
from ._common import BaseSVR

from sklearn.svm import NuSVR as sklearn_NuSVR
from sklearn.utils.validation import _deprecate_positional_args

from onedal.svm import NuSVR as onedal_NuSVR


class NuSVR(sklearn_NuSVR, BaseSVR):
    @_deprecate_positional_args
    def __init__(self, *, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, nu=0.5, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, nu=nu,
            shrinking=shrinking, cache_size=cache_size, verbose=verbose,
            max_iter=max_iter)

    def fit(self, X, y, sample_weight=None):
        if self.kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
            logging.info("sklearn.svm.NuSVR.fit: " + get_patch_message("onedal"))
            self._onedal_fit(X, y, sample_weight)
        else:
            logging.info("sklearn.svm.NuSVR.fit: " + get_patch_message("sklearn"))
            sklearn_NuSVR.fit(self, X, y, sample_weight)
        return self

    def predict(self, X):
        if hasattr(self, '_onedal_estimator'):
            logging.info("sklearn.svm.NuSVR.predict: " + get_patch_message("onedal"))
            return self._onedal_estimator.predict(X)
        else:
            logging.info("sklearn.svm.NuSVR.predict: " + get_patch_message("sklearn"))
            return sklearn_NuSVR.predict(self, X)

    def _onedal_fit(self, X, y, sample_weight=None):
        onedal_params = {
            'C': self.C,
            'nu': self.nu,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
        }

        self._onedal_estimator = onedal_NuSVR(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight)
        self._save_attributes()
