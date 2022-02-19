#!/usr/bin/env python
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

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
import warnings

from sklearn.neighbors._base import NeighborsBase as sklearn_NeighborsBase
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree
from sklearn.neighbors._base import _check_weights
from sklearn.neighbors._base import VALID_METRICS
from sklearn.neighbors._classification import KNeighborsClassifier as \
    sklearn_KNeighborsClassifier
from sklearn.neighbors._unsupervised import NearestNeighbors as \
    sklearn_NearestNeighbors
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from onedal.datatypes import _check_array, _num_features, _num_samples
from onedal.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier

from .._device_offload import dispatch, wrap_output_data
import numpy as np
from scipy import sparse as sp


if Version(sklearn_version) >= Version("0.24"):
    class KNeighborsClassifier_(sklearn_KNeighborsClassifier):
        @_deprecate_positional_args
        def __init__(self, n_neighbors=5, *,
                     weights='uniform', algorithm='auto', leaf_size=30,
                     p=2, metric='minkowski', metric_params=None, n_jobs=None,
                     **kwargs):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params,
                n_jobs=n_jobs, **kwargs)
            self.weights = \
                weights if Version(sklearn_version) >= Version("1.0") \
                else _check_weights(weights)
elif Version(sklearn_version) >= Version("0.22"):
    from sklearn.neighbors._base import SupervisedIntegerMixin as \
        BaseSupervisedIntegerMixin

    class KNeighborsClassifier_(sklearn_KNeighborsClassifier,
                                BaseSupervisedIntegerMixin):
        @_deprecate_positional_args
        def __init__(self, n_neighbors=5, *,
                     weights='uniform', algorithm='auto', leaf_size=30,
                     p=2, metric='minkowski', metric_params=None, n_jobs=None,
                     **kwargs):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params,
                n_jobs=n_jobs, **kwargs)
            self.weights = _check_weights(weights)
else:
    from sklearn.neighbors.base import SupervisedIntegerMixin as \
        BaseSupervisedIntegerMixin

    class KNeighborsClassifier_(sklearn_KNeighborsClassifier,
                                BaseSupervisedIntegerMixin):
        @_deprecate_positional_args
        def __init__(self, n_neighbors=5, *,
                     weights='uniform', algorithm='auto', leaf_size=30,
                     p=2, metric='minkowski', metric_params=None, n_jobs=None,
                     **kwargs):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params,
                n_jobs=n_jobs, **kwargs)
            self.weights = _check_weights(weights)


class KNeighborsClassifier(KNeighborsClassifier_):
    @_deprecate_positional_args
    def __init__(self, n_neighbors=5, *,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params,
            n_jobs=n_jobs, **kwargs)

    def fit(self, X, y):
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=True)
        if self.metric_params is not None and 'p' in self.metric_params:
            if self.p is not None:
                warnings.warn("Parameter p is found in metric_params. "
                              "The corresponding parameter from __init__ "
                              "is ignored.", SyntaxWarning, stacklevel=2)
            self.effective_metric_params_ = self.metric_params.copy()
            effective_p = self.metric_params["p"]
        else:
            self.effective_metric_params_ = {}
            effective_p = self.p

        if self.metric in ["minkowski"]:
            if effective_p < 1:
                raise ValueError("p must be greater or equal to one for minkowski metric")
            self.effective_metric_params_["p"] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == "minkowski":
            p = self.effective_metric_params_.pop("p", 2)
            if p < 1:
                raise ValueError(
                    "p must be greater or equal to one for minkowski metric"
                )
            if p == 1:
                self.effective_metric_ = "manhattan"
            elif p == 2:
                self.effective_metric_ = "euclidean"
            elif p == np.inf:
                self.effective_metric_ = "chebyshev"
            else:
                self.effective_metric_params_["p"] = p

        if self.metric == "manhattan":
            self.p = 1

        if not isinstance(X, (KDTree, BallTree, sklearn_NeighborsBase)):
            self._fit_X = _check_array(
                X, dtype=[np.float64, np.float32], accept_sparse=True)
            self.n_samples_fit_ = _num_samples(self._fit_X)
            self.n_features_in_ = _num_features(self._fit_X)

            if self.algorithm == "auto":
                # A tree approach is better for small number of neighbors or small
                # number of features, with KDTree generally faster when available
                is_n_neighbors_valid_for_brute = self.n_neighbors is not None and \
                    self.n_neighbors >= self._fit_X.shape[0] // 2
                if self._fit_X.shape[1] > 15 or is_n_neighbors_valid_for_brute:
                    self._fit_method = "brute"
                else:
                    if self.effective_metric_ in VALID_METRICS["kd_tree"]:
                        self._fit_method = "kd_tree"
                    elif callable(self.effective_metric_) or \
                        self.effective_metric_ in \
                            VALID_METRICS["ball_tree"]:
                        self._fit_method = "ball_tree"
                    else:
                        self._fit_method = "brute"
            else:
                self._fit_method = self.algorithm

        if hasattr(self, '_onedal_estimator'):
            delattr(self, '_onedal_estimator')
        # To cover test case when we pass patched
        # estimator as an input for other estimator
        if isinstance(X, sklearn_NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            self.n_samples_fit_ = X.n_samples_fit_
            self.n_features_in_ = X.n_features_in_
            if hasattr(X, '_onedal_estimator'):
                if self._fit_method == "ball_tree":
                    X._tree = BallTree(
                        X._fit_X,
                        self.leaf_size,
                        metric=self.effective_metric_,
                        **self.effective_metric_params_,
                    )
                elif self._fit_method == "kd_tree":
                    X._tree = KDTree(
                        X._fit_X,
                        self.leaf_size,
                        metric=self.effective_metric_,
                        **self.effective_metric_params_,
                    )
                elif self._fit_method == "brute":
                    X._tree = None
                else:
                    raise ValueError("algorithm = '%s' not recognized" % self.algorithm)

        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'ball_tree'
            self.n_samples_fit_ = X.data.shape[0]
            self.n_features_in_ = X.data.shape[1]

        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'kd_tree'
            self.n_samples_fit_ = X.data.shape[0]
            self.n_features_in_ = X.data.shape[1]

        dispatch(self, 'neighbors.KNeighborsClassifier.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_KNeighborsClassifier.fit,
        }, X, y)
        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'neighbors.KNeighborsClassifier.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_KNeighborsClassifier.predict,
        }, X)

    @wrap_output_data
    def predict_proba(self, X):
        check_is_fitted(self)
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'neighbors.KNeighborsClassifier.predict_proba', {
            'onedal': self.__class__._onedal_predict_proba,
            'sklearn': sklearn_KNeighborsClassifier.predict_proba,
        }, X)

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'neighbors.KNeighborsClassifier.kneighbors', {
            'onedal': self.__class__._onedal_kneighbors,
            'sklearn': sklearn_KNeighborsClassifier.kneighbors,
        }, X, n_neighbors, return_distance)

    @wrap_output_data
    def radius_neighbors(self, X=None, radius=None, return_distance=True,
                         sort_results=False):
        _onedal_estimator = getattr(self, '_onedal_estimator', None)

        if _onedal_estimator is not None or getattr(self, '_tree', 0) is None and \
                self._fit_method == 'kd_tree':
            if Version(sklearn_version) >= Version("0.24"):
                sklearn_NearestNeighbors.fit(self, self._fit_X, getattr(self, '_y', None))
            else:
                sklearn_NearestNeighbors.fit(self, self._fit_X)
        if Version(sklearn_version) >= Version("0.22"):
            result = sklearn_NearestNeighbors.radius_neighbors(
                self, X, radius, return_distance, sort_results)
        else:
            result = sklearn_NearestNeighbors.radius_neighbors(
                self, X, radius, return_distance)

        return result

    def _onedal_gpu_supported(self, method_name, *data):
        X_incorrect_type = isinstance(data[0], (KDTree, BallTree, sklearn_NeighborsBase))

        if X_incorrect_type:
            return False

        if self._fit_method in ['auto', 'ball_tree']:
            condition = self.n_neighbors is not None and \
                self.n_neighbors >= self.n_samples_fit_ // 2
            if self.n_features_in_ > 15 or condition:
                result_method = 'brute'
            else:
                if self.effective_metric_ in ['euclidean']:
                    result_method = 'kd_tree'
                else:
                    result_method = 'brute'
        else:
            result_method = self._fit_method

        is_sparse = sp.isspmatrix(data[0])
        is_single_output = False
        class_count = 1
        if len(data) > 1 or hasattr(self, '_onedal_estimator'):
            # To check multioutput, might be overhead
            if len(data) > 1:
                y = np.asarray(data[1])
                class_count = len(np.unique(y))
            if hasattr(self, '_onedal_estimator'):
                y = self._onedal_estimator._y
            is_single_output = y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1
        is_valid_for_brute = result_method in ['brute'] and \
            self.effective_metric_ in ['manhattan',
                                       'minkowski',
                                       'euclidean',
                                       'chebyshev',
                                       'cosine']
        is_valid_weights = self.weights in ['uniform', "distance"]
        main_condition = is_valid_for_brute and not is_sparse and \
            is_single_output and is_valid_weights

        if method_name == 'neighbors.KNeighborsClassifier.fit':
            return main_condition and class_count >= 2
        if method_name in ['neighbors.KNeighborsClassifier.predict',
                           'neighbors.KNeighborsClassifier.predict_proba',
                           'neighbors.KNeighborsClassifier.kneighbors']:
            return main_condition and hasattr(self, '_onedal_estimator')
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_cpu_supported(self, method_name, *data):
        X_incorrect_type = isinstance(data[0], (KDTree, BallTree, sklearn_NeighborsBase))

        if X_incorrect_type:
            return False

        if self._fit_method in ['auto', 'ball_tree']:
            condition = self.n_neighbors is not None and \
                self.n_neighbors >= self.n_samples_fit_ // 2
            if self.n_features_in_ > 15 or condition:
                result_method = 'brute'
            else:
                if self.effective_metric_ in ['euclidean']:
                    result_method = 'kd_tree'
                else:
                    result_method = 'brute'
        else:
            result_method = self._fit_method

        is_sparse = sp.isspmatrix(data[0])
        is_single_output = False
        class_count = 1
        if len(data) > 1 or hasattr(self, '_onedal_estimator'):
            # To check multioutput, might be overhead
            if len(data) > 1:
                y = np.asarray(data[1])
                class_count = len(np.unique(y))
            if hasattr(self, '_onedal_estimator'):
                y = self._onedal_estimator._y
            is_single_output = y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1
        is_valid_for_kd_tree = \
            result_method in ['kd_tree'] and self.effective_metric_ in ['euclidean']
        is_valid_for_brute = result_method in ['brute'] and \
            self.effective_metric_ in ['manhattan',
                                       'minkowski',
                                       'euclidean',
                                       'chebyshev',
                                       'cosine']
        is_valid_weights = self.weights in ['uniform', "distance"]
        main_condition = (is_valid_for_kd_tree or is_valid_for_brute) and \
            not is_sparse and is_single_output and is_valid_weights

        if method_name == 'neighbors.KNeighborsClassifier.fit':
            return main_condition and class_count >= 2
        if method_name in ['neighbors.KNeighborsClassifier.predict',
                           'neighbors.KNeighborsClassifier.predict_proba',
                           'neighbors.KNeighborsClassifier.kneighbors']:
            return main_condition and hasattr(self, '_onedal_estimator')
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y, queue=None):
        onedal_params = {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.effective_metric_,
            'p': self.p,
        }

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        self._onedal_estimator = onedal_KNeighborsClassifier(**onedal_params)
        self._onedal_estimator.requires_y = requires_y
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.fit(X, y, queue=queue)

        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_predict_proba(self, X, queue=None):
        return self._onedal_estimator.predict_proba(X, queue=queue)

    def _onedal_kneighbors(self, X=None, n_neighbors=None,
                           return_distance=True, queue=None):
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue)

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._y = self._onedal_estimator._y
        self._fit_method = self._onedal_estimator._fit_method
        self.outputs_2d_ = self._onedal_estimator.outputs_2d_
        self._tree = self._onedal_estimator._tree
