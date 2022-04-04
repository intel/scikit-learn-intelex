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

import numpy as np
import numbers

from .._device_offload import dispatch, wrap_output_data
from sklearn.utils.validation import _deprecate_positional_args, _num_samples
from onedal.kmeans import KMeans as onedal_KMeans

from sklearn.cluster import KMeans as sklearn_KMeans
from sklearn import __version__ as sklearn_version

class KMeans(sklearn_KMeans):
    __doc__ = sklearn_KMeans.__doc__

    @_deprecate_positional_args
    def __init__(self, n_clusters=8, *, init='k-means++', n_init=10,
                 max_iter=300, tol=0.0001, verbose=0, random_state=None,
                 copy_x=True, algorithm='auto'):
        super().__init__(
            n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
            tol=tol, verbose=verbose, random_state=random_state, copy_x=copy_x,
            algorithm=algorithm)

    def fit(self, X, y=None, sample_weight=None):
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=True)
        dispatch(self, 'kmeans.KMeans.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_KMeans.fit,
        }, X, y, sample_weight)
        return self
    
    @wrap_output_data
    def predict(self, X):
        if Version(sklearn_version) >= Version("1.0"):
            self._check_feature_names(X, reset=False)
        return dispatch(self, 'kmeans.KMeans.predict', {
            'onedal': self.__class__._onedal_predict,
            'sklearn': sklearn_KMeans.predict,
        }, X)

    def _onedal_gpu_supported(self, method_name, *data):
        if method_name == 'kmeans.KMeans.fit':
            return True #TODO check this
        if method_name == 'kmeans.KMeans.predict':
            return self._onedal_gpu_supported('kmeans.KMeans.fit', *data)
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_cpu_supported(self, method_name, *data):
        if method_name == 'kmeans.KMeans.fit':
            sample_weight = data[2]
            X_len = _num_samples(data[0])
            if sample_weight is not None:
                if isinstance(sample_weight, numbers.Number):
                    sample_weight = np.full(X_len, sample_weight, dtype=np.float64)
                else:
                    sample_weight = np.asarray(sample_weight)
                return sample_weight.shape == (X_len,) and np.allclose(sample_weight, np.ones_like(sample_weight))
            return True #TODO check this
        if method_name == 'kmeans.KMeans.predict':
            return hasattr(self, '_onedal_estimator')
        raise RuntimeError(f'Unknown method {method_name} in {self.__class__.__name__}')

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        onedal_params = {
            'n_clusters': self.n_clusters,
            'init': self.init,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'copy_x': self.copy_x,
            'algorithm': self.algorithm
        }
        
        self._onedal_estimator = onedal_KMeans(**onedal_params)
        self._onedal_estimator.fit(X, y, sample_weight, queue=queue)

        self._save_attributes()
    
    def _onedal_predict(self, X, queue=None):
        return self._onedal_estimator.predict(X, queue=queue)

    def _save_attributes(self):
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.labels_ = self._onedal_estimator.labels_
        self.cluster_centers_ = self._onedal_estimator.cluster_centers_
        self.n_iter_ = self._onedal_estimator.n_iter_
        self.inertia_ = self._onedal_estimator.inertia_