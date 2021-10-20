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

from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version

from sklearn.neighbors._unsupervised import NearestNeighbors as \
    sklearn_NearestNeighbors

from sklearn.utils.validation import _deprecate_positional_args

from onedal.datatypes import _check_array

from onedal.neighbors import NearestNeighbors as onedal_NearestNeighbors

from .._device_offload import dispatch, wrap_output_data
import numpy as np
from scipy import sparse as sp


if LooseVersion(sklearn_version) >= LooseVersion("0.22") and \
    LooseVersion(sklearn_version) < LooseVersion("0.23"):
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


class NearestNeighbors(NearestNeighbors_):
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
        if hasattr(self, '_onedal_estimator'):
            delattr(self, '_onedal_estimator')

        dispatch(self, 'neighbors.NearestNeighbors.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_NearestNeighbors.fit,
        }, X, None)
        return self

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        return dispatch(self, 'neighbors.NearestNeighbors.kneighbors', {
            'onedal': self.__class__._onedal_kneighbors,
            'sklearn': sklearn_NearestNeighbors.kneighbors,
        }, X, n_neighbors, return_distance)


    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'neighbors.NearestNeighbors.fit':
            is_sparse = sp.isspmatrix(data[0])
            class_count = None
            if len(data) > 1:
                class_count = len(np.unique(data[1]))
            return self.weights in ['uniform', 'distance'] and \
                self.algorithm in ['brute', 'auto'] and \
                self.metric in ['minkowski', 'euclidean'] and \
                self.class_weight is None and \
                class_count >= 2 and \
                not is_sparse
        if method_name in ['neighbors.NearestNeighbors.kneighbors']:
            return hasattr(self, '_onedal_estimator') and not sp.isspmatrix(data[0])
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'neighbors.NearestNeighbors.fit':
            is_sparse = sp.isspmatrix(data[0])
            class_count = None
            if len(data) > 1:
                class_count = len(np.unique(data[1]))
            return self.weights in ['uniform', 'distance'] \
                    and self.algorithm in ['brute', 'kd_tree', 'auto', 'ball_tree'] \
                    and self.metric in ['minkowski', 'euclidean', 'chebyshev', 'cosine'] \
                    and class_count >= 2 \
                    and not is_sparse
        if method_name in ['neighbors.NearestNeighbors.kneighbors']:
            print("_onedal_estimator ", hasattr(self, '_onedal_estimator'))
            print("sparse ", not sp.isspmatrix(data[0]))
            return hasattr(self, '_onedal_estimator') and not sp.isspmatrix(data[0])
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y=None, queue=None):
        onedal_params = {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'p': self.p,
            'metric_params': self.metric_params,
        }

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        self._onedal_estimator = onedal_NearestNeighbors(**onedal_params)
        self._onedal_estimator.requires_y = requires_y
        self._onedal_estimator.fit(X, y, queue=queue)

        self._save_attributes()

    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return self._onedal_estimator.kneighbors(X, n_neighbors, return_distance, queue=queue)

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._y = self._onedal_estimator._y
        self.shape = self._onedal_estimator.shape
        self.effective_metric_ = self._onedal_estimator.effective_metric_
        self.effective_metric_params_ = self._onedal_estimator.effective_metric_params_
        self._fit_method = self._onedal_estimator._fit_method
        self.outputs_2d_ = self._onedal_estimator.outputs_2d_
        self._tree = self._onedal_estimator._tree

