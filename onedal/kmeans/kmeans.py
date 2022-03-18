#===============================================================================
# Copyright 2022 Intel Corporation
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

from abc import ABCMeta

from numbers import Integral

import numpy as np
from ..datatypes import (
    _check_X_y,
    _check_array,
    _column_or_1d,
    _check_n_features,
    _check_classification_targets,
    _num_samples
)

from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes._data_conversion import from_table, to_table

def _is_arraylike_not_scalar(x): # import from sklearn????? here????
    return (hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")) and not np.isscalar(x)

class KMeans(metaclass=ABCMeta):
    def __init__(self, n_clusters=8, *, init='k-means++',
                 n_init=10, max_iter=300, tol=0.0001,
                 verbose=0, random_state=None, copy_x=True,
                 algorithm='auto'):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _get_onedal_params(self, data):
        return {
            'fptype': 'float' if data.dtype is np.dtype('float32') else 'double',
            'method': 'lloyd_dense', # hardcode this???
            'cluster_count': self.n_clusters,
            'max_iteration_count': self.max_iter,
            'accuracy_threshold': self.tol
        }
    
    def _get_init_onedal_params(self, data):
        str: init_method
        if (self.init == 'k-means++'):
            init_method = "plus_plus_dense"
        elif (self.init == 'random'):
            init_method = "random_dense"
        else:
            raise ValueError("Wrong init str")
        return {
            'fptype': 'float' if data.dtype is np.dtype('float32') else 'double',
            'method': init_method,
            'cluster_count': self.n_clusters
        }
    
    def _init_centroids_onedal(self, X, queue):
        module = _backend.kmeans_init.init
        policy = _get_policy(queue, X)
        params = self._get_init_onedal_params(X)
        result = module.compute(policy, params, to_table(X)) # extra convert to table???
        return result.centroids

    def _init_centroids(self, X, queue):
        if isinstance(self.init, str):
            centers = self._init_centroids_onedal(X, queue)
        elif _is_arraylike_not_scalar(self.init):
            centers = to_table(self.init) # should convert here?
        elif callable(self.init):
            centers = to_table(init(X, self.n_clusters, random_state=self.random_state))
            #some checks in stock
        return centers

    def fit(self, X, y=None, sample_weight=None, queue=None):
        module = _backend.kmeans.clustering

        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)
        centroids = self._init_centroids(X, queue)
        result = module.train(policy, params, to_table(X), centroids)

        self.cluster_centers_ = result.model.centroids
        self.labels_ = result.responses # wrong format
        self.inertia_ = result.objective_function_value # wrong format
        self.n_iter_ = result.iteration_count # wrong format
#        self.n_features_in_ = # new in 0.24
#        self.feature_names_in_ = # new in 1.0
        return result
