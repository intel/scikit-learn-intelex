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
from daal4py.sklearn._utils import sklearn_check_version
import warnings

from sklearn.neighbors._base import NeighborsBase as sklearn_NeighborsBase
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree
from sklearn.neighbors._base import VALID_METRICS
from sklearn.neighbors._unsupervised import NearestNeighbors as \
    sklearn_NearestNeighbors

from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted

from onedal.datatypes import _check_array, _num_features, _num_samples
from onedal.neighbors import NearestNeighbors as onedal_NearestNeighbors

from .common import KNeighborsDispatchingBase
from .._device_offload import dispatch, wrap_output_data
import numpy as np


if sklearn_check_version("0.22") and \
   Version(sklearn_version) < Version("0.23"):
    class NearestNeighbors_(sklearn_NearestNeighbors):
        def __init__(self, n_neighbors=5, radius=1.0,
                     algorithm='auto', leaf_size=30, metric='minkowski',
                     p=2, metric_params=None, n_jobs=None):
            super().__init__(
                n_neighbors=n_neighbors,
                radius=radius,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params, n_jobs=n_jobs)
else:
    class NearestNeighbors_(sklearn_NearestNeighbors):
        if sklearn_check_version('1.2'):
            _parameter_constraints: dict = {
                **sklearn_NearestNeighbors._parameter_constraints}

        @_deprecate_positional_args
        def __init__(self, *, n_neighbors=5, radius=1.0,
                     algorithm='auto', leaf_size=30, metric='minkowski',
                     p=2, metric_params=None, n_jobs=None):
            super().__init__(
                n_neighbors=n_neighbors,
                radius=radius,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params, n_jobs=n_jobs)


class NearestNeighbors(NearestNeighbors_, KNeighborsDispatchingBase):
    if sklearn_check_version('1.2'):
        _parameter_constraints: dict = {
            **NearestNeighbors_._parameter_constraints}

    @_deprecate_positional_args
    def __init__(self, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=None):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, X, y=None):
        self._fit_validation(X, y)
        dispatch(self, 'fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_NearestNeighbors.fit,
        }, X, None)
        return self

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if sklearn_check_version("1.0") and X is not None:
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'kneighbors', {
            'onedal': self.__class__._onedal_kneighbors,
            'sklearn': sklearn_NearestNeighbors.kneighbors,
        }, X, n_neighbors, return_distance)

    @wrap_output_data
    def radius_neighbors(self, X=None, radius=None, return_distance=True,
                         sort_results=False):
        _onedal_estimator = getattr(self, '_onedal_estimator', None)

        if _onedal_estimator is not None or getattr(self, '_tree', 0) is None and \
                self._fit_method == 'kd_tree':
            if sklearn_check_version("0.24"):
                sklearn_NearestNeighbors.fit(self, self._fit_X, getattr(self, '_y', None))
            else:
                sklearn_NearestNeighbors.fit(self, self._fit_X)
        if sklearn_check_version("0.22"):
            result = sklearn_NearestNeighbors.radius_neighbors(
                self, X, radius, return_distance, sort_results)
        else:
            result = sklearn_NearestNeighbors.radius_neighbors(
                self, X, radius, return_distance)

        return result

    def _onedal_fit(self, X, y=None, queue=None):
        onedal_params = {
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'metric': self.effective_metric_,
            'p': self.effective_metric_params_['p'],
        }

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        self._onedal_estimator = onedal_NearestNeighbors(**onedal_params)
        self._onedal_estimator.requires_y = requires_y
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.fit(X, y, queue=queue)

        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_kneighbors(self, X=None, n_neighbors=None,
                           return_distance=True, queue=None):
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue)

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._fit_method = self._onedal_estimator._fit_method
        self._tree = self._onedal_estimator._tree
