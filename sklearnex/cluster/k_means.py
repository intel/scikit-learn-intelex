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

from .._device_offload import dispatch

from sklearn.utils.validation import _deprecate_positional_args

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
        # if Version(sklearn_version) >= Version("1.0"):
        #     self._check_feature_names(X, reset=True)
        dispatch(self, 'kmeans.KMeans.fit', {
            'onedal': self.__class__._onedal_fit,
            'sklearn': sklearn_KMeans.fit,
        }, X, y, sample_weight)
        return self
    
    @wrap_output_data
    def predict(self, X):
        # if Version(sklearn_version) >= Version("1.0"):
        #     self._check_feature_names(X, reset=False)
        # return dispatch(self, 'svm.SVC.predict', {
        #     'onedal': self.__class__._onedal_predict,
        #     'sklearn': sklearn_SVC.predict,
        # }, X)
        raise "Unimplemented"

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
