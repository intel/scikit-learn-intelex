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
from ._common import BaseSVR
from .._device_offload import _dispatch

from sklearn.svm import SVR as sklearn_SVR
from sklearn.utils.validation import _deprecate_positional_args

from onedal.svm import SVR as onedal_SVR


class SVR(sklearn_SVR, BaseSVR):
    @_deprecate_positional_args
    def __init__(self, *, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C,
            epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,
            max_iter=max_iter)

    def _gpu_supported(self, method_name, *data):
        return False

    def _cpu_supported(self, method_name, *data):
        if method_name == 'svm.SVC.fit':
            return self.kernel in ['linear', 'rbf', 'poly', 'sigmoid']
        if method_name in ['svm.SVC.predict',
                           'svm.SVC._predict_proba',
                           'svm.SVC.decision_function']:
            return hasattr(self, '_onedal_estimator')

    def fit(self, X, y, sample_weight=None):
        _dispatch(self, 'svm.SVR.fit', {
            'onedal': lambda q, X, y, w: self._onedal_fit(X, y, sample_weight=w, queue=q),
            'sklearn': lambda X, y, w: sklearn_SVR.fit(self, X, y, sample_weight=w),
        }, X, y, sample_weight)

        return self

    def predict(self, X):
        return _dispatch(self, 'svm.SVR.predict', {
            'onedal': lambda q, X: self._onedal_estimator.predict(X, queue=q),
            'sklearn': lambda X: sklearn_SVR.predict(self, X),
        }, X)

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        onedal_params = {
            'C': self.C,
            'epsilon': self.epsilon,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
        }

        self._onedal_estimator = onedal_SVR(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)
        self._save_attributes()
