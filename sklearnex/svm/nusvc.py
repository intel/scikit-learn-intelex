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
        dispatch(self, 'svm.NuSVC.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_NuSVC.fit,
        }, X, y, sample_weight)

        return self

    @wrap_output_data
    def predict(self, X):
        return dispatch(self, 'svm.NuSVC.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_NuSVC.predict,
        }, X)

    @wrap_output_data
    def _predict_proba(self, X):
        return dispatch(self, 'svm.NuSVC._predict_proba', {
            'onedal': self.__class__._onedal_predict_proba,
            'sklearn': sklearn_NuSVC._predict_proba,
        }, X)

    @wrap_output_data
    def decision_function(self, X):
        return dispatch(self, 'svm.NuSVC.decision_function', {
            'onedal': self.__class__._onedal_decision_function,
            'sklearn': sklearn_NuSVC.decision_function,
        }, X)

    def _onedal_gpu_supported(self, method_name, *data):
        return False

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'svm.NuSVC.fit':
            return self.kernel in ['linear', 'rbf', 'poly', 'sigmoid']
        if method_name in ['svm.NuSVC.predict',
                           'svm.NuSVC._predict_proba',
                           'svm.NuSVC.decision_function']:
            return hasattr(self, '_onedal_estimator')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
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
