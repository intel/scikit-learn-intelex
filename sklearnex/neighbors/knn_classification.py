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

from sklearn.neighbors._base import KNeighborsMixin as sklearn_KNeighborsMixin
from sklearn.neighbors._base import RadiusNeighborsMixin as sklearn_RadiusNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as sklearn_NeighborsBase
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree
from sklearn.neighbors._base import _check_weights

from sklearn.neighbors._classification import KNeighborsClassifier as \
    sklearn_KNeighborsClassifier

from sklearn.utils.validation import _deprecate_positional_args

from onedal.datatypes import _check_array

from onedal.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier
from onedal.neighbors import KNeighborsMixin as onedal_KNeighborsMixin

from .._device_offload import dispatch, wrap_output_data
import numpy as np
from scipy import sparse as sp


class KNeighborsMixin(sklearn_KNeighborsMixin):
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        onedal_model = getattr(self, '_onedal_model', None)
        if X is not None:
            X = _check_array(
                X, accept_sparse='csr', dtype=[
                    np.float64, np.float32])

        if onedal_model is not None and not sp.issparse(X):
            result = onedal_KNeighborsMixin._kneighbors(self, X, n_neighbors, return_distance)

        return result

if LooseVersion(sklearn_version) >= LooseVersion("0.24"):
    class KNeighborsClassifier_(sklearn_NeighborsBase, KNeighborsMixin, onedal_KNeighborsClassifier):
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
                weights if LooseVersion(sklearn_version) >= LooseVersion("1.0") else _check_weights(weights)
elif LooseVersion(sklearn_version) >= LooseVersion("0.22"):
    from sklearn.neighbors._base import SupervisedIntegerMixin as \
        BaseSupervisedIntegerMixin

    class KNeighborsClassifier_(sklearn_NeighborsBase, onedal_KNeighborsClassifier, KNeighborsMixin,
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

    class KNeighborsClassifier_(sklearn_NeighborsBase, onedal_KNeighborsClassifier, KNeighborsMixin,
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
        return dispatch(self, 'neighbors.KNeighborsClassifier.fit', {
            'onedal': onedal_KNeighborsClassifier.fit,
            'sklearn': sklearn_KNeighborsClassifier.fit,
        }, X, y)

    @wrap_output_data
    def predict(self, X):
        return dispatch(self, 'neighbors.KNeighborsClassifier.predict', {
            'onedal': onedal_KNeighborsClassifier.predict,
            'sklearn': sklearn_KNeighborsClassifier.predict,
        }, X)

    @wrap_output_data
    def predict_proba(self, X):
        return dispatch(self, 'neighbors.KNeighborsClassifier.predict_proba', {
            # 'onedal': onedal_KNeighborsClassifier.predict,
            'sklearn': sklearn_KNeighborsClassifier.predict_proba,
        }, X)

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'neighbors.KNeighborsClassifier.fit':
            if len(data) > 1:
                import numpy as np
                from scipy import sparse as sp

                class_count = len(np.unique(data[1]))
                is_sparse = sp.isspmatrix(data[0])
            return self.weights in ['uniform', 'distance'] and \
                self.algorithm in ['brute', 'auto'] and \
                self.metric in ['minkowski', 'euclidean'] and \
                self.class_weight is None and \
                class_count >= 2 and \
                not is_sparse
        if method_name in ['neighbors.KNeighborsClassifier.predict',
                           'neighbors.KNeighborsClassifier.predict_proba']:
            return hasattr(self, '_onedal_model') and not sp.isspmatrix(data[0])
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'neighbors.KNeighborsClassifier.fit':
            return self.weights in ['uniform', 'distance'] \
                    and self.algorithm in ['brute', 'kd_tree', 'auto', 'ball_tree'] \
                    and self.metric in ['minkowski', 'euclidean', 'chebyshev', 'cosine']
        if method_name in ['neighbors.KNeighborsClassifier.predict',
                           'neighbors.KNeighborsClassifier.predict_proba']:
            return hasattr(self, '_onedal_model') and not sp.isspmatrix(data[0])
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

