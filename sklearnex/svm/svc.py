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

from ._common import BaseSVC
from .._device_offload import dispatch, wrap_output_data

from sklearn.svm import SVC as sklearn_SVC
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.exceptions import NotFittedError
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion

from onedal.svm import SVC as onedal_SVC


class SVC(sklearn_SVC, BaseSVC):
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
        if LooseVersion(sklearn_version) >= LooseVersion("1.0"):
            self._check_feature_names(X, reset=True)
        dispatch(self, 'svm.SVC.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_SVC.fit,
        }, X, y, sample_weight)
        return self

    @wrap_output_data
    def predict(self, X):
        if LooseVersion(sklearn_version) >= LooseVersion("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'svm.SVC.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_SVC.predict,
        }, X)

    @property
    def predict_proba(self):
        self._check_proba()
        return self._predict_proba

    @wrap_output_data
    def _predict_proba(self, X):
        sklearn_pred_proba = (sklearn_SVC.predict_proba
                              if LooseVersion(sklearn_version) >= LooseVersion("1.0")
                              else sklearn_SVC._predict_proba)

        return dispatch(self, 'svm.SVC.predict_proba', {
            'onedal': self.__class__._onedal_predict_proba,
            'sklearn': sklearn_pred_proba,
        }, X)

    @wrap_output_data
    def decision_function(self, X):
        if LooseVersion(sklearn_version) >= LooseVersion("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'svm.SVC.decision_function', {
            'onedal': self.__class__._onedal_decision_function,
            'sklearn': sklearn_SVC.decision_function,
        }, X)

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'svm.SVC.fit':
            if len(data) > 1:
                import numpy as np
                from scipy import sparse as sp

                self._class_count = len(np.unique(data[1]))
                self._is_sparse = sp.isspmatrix(data[0])
            return self.kernel in ['linear', 'rbf'] and \
                self.class_weight is None and \
                hasattr(self, '_class_count') and self._class_count == 2 and \
                hasattr(self, '_is_sparse') and not self._is_sparse
        if method_name in ['svm.SVC.predict',
                           'svm.SVC.predict_proba',
                           'svm.SVC.decision_function']:
            return hasattr(self, '_onedal_estimator') and \
                self._onedal_gpu_supported('svm.SVC.fit', *data)
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'svm.SVC.fit':
            return self.kernel in ['linear', 'rbf', 'poly', 'sigmoid']
        if method_name in ['svm.SVC.predict',
                           'svm.SVC.predict_proba',
                           'svm.SVC.decision_function']:
            return hasattr(self, '_onedal_estimator')
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
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
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        if self.class_weight == 'balanced':
            self.class_weight_ = self._compute_balanced_class_weight(y)
        else:
            self.class_weight_ = self._onedal_estimator.class_weight_

        if self.probability:
            self._fit_proba(X, y, sample_weight, queue=queue)
        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_predict_proba(self, X, queue=None):
        if getattr(self, 'clf_prob', None) is None:
            raise NotFittedError(
                "predict_proba is not available when fitted with probability=False")
        from .._config import get_config, config_context

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg['target_offload'] = queue
        with config_context(**cfg):
            return self.clf_prob.predict_proba(X)

    def _onedal_decision_function(self, X, queue=None):
        return self._onedal_estimator.decision_function(X, queue=queue)
