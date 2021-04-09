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
from ._common import get_dual_coef, set_dual_coef, get_intercept, set_intercept

from sklearn.svm import SVR as sklearn_SVR
from sklearn.utils.validation import _deprecate_positional_args

from onedal.svm import SVR as onedal_SVR


class SVR(sklearn_SVR):
    @_deprecate_positional_args
    def __init__(self, *, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C,
            epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,
            max_iter=max_iter)

    def _create_onedal_svr(self):
        return onedal_SVR(C=self.C, epsilon=self.epsilon,
                          kernel=self.kernel, degree=self.degree,
                          gamma=self.gamma, coef0=self.coef0,
                          tol=self.tol, shrinking=self.shrinking,
                          cache_size=self.cache_size,
                          max_iter=self.max_iter)

    def fit(self, X, y, sample_weight=None):
        if self.kernel in ['linear', 'rbf', 'poly'] and not sp.isspmatrix(X):
            logging.info("sklearn.svm.SVR.fit: " + get_patch_message("onedal"))

            self._onedal_model = self._create_onedal_svr()
            self._onedal_model.fit(X, y, sample_weight)

            self.support_vectors_ = self._onedal_model.support_vectors_
            self.n_features_in_ = self._onedal_model.n_features_in_
            self.fit_status_ = 0
            self.dual_coef_ = self._onedal_model.dual_coef_
            self.shape_fit_ = self._onedal_model.shape_fit_
            self.support_ = self._onedal_model.support_

            self._intercept_ = self._onedal_model.intercept_
            self._n_support = [self.support_vectors_.shape[0]]
            self._sparse = False
            self._gamma = self._onedal_model._gamma
            self._probA = None
            self._probB = None

            self._dual_coef_ = property(get_dual_coef, set_dual_coef)
            self.intercept_ = property(get_intercept, set_intercept)

            self._is_in_fit = True
            self._dual_coef_ = self.dual_coef_
            self.intercept_ = self._intercept_
            self._is_in_fit = False
        else:
            logging.info("sklearn.svm.SVR.fit: " + get_patch_message("sklearn"))
            sklearn_SVR.fit(self, X, y, sample_weight)

        return self

    def predict(self, X):
        if hasattr(self, '_onedal_model') and not sp.isspmatrix(X):
            logging.info("sklearn.svm.SVR.predict: " + get_patch_message("onedal"))
            return self._onedal_model.predict(X)
        else:
            logging.info("sklearn.svm.SVR.predict: " + get_patch_message("sklearn"))
            return sklearn_SVR.predict(self, X)
