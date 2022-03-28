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
from ..common._estimator_checks import _check_is_fitted

import numpy as np
from ..datatypes import (
    _check_array,
    _check_random_state,
    _is_arraylike_not_scalar
)

from onedal import _backend

from ..common._policy import _get_policy
from ..datatypes._data_conversion import from_table, to_table

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

    def _tolerance(self, X, rtol):
        """Compute absolute tolerance from the relative tolerance"""
        if rtol == 0.0:
            return rtol
        mean_var = np.var(X, axis=0).mean()
        return mean_var * rtol

    def _check_test_data(self, X):
        X = _check_array(
            X,
            accept_sparse='csr',
            dtype=[np.float64, np.float32],
            accept_large_sparse=False
        )
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                (f'X has {X.shape[1]} features, '
                f'but Kmeans is expecting {self.n_features_in_} features as input'))
        return X

    def _get_onedal_params(self, data):
        return {
            'fptype': 'float' if data.dtype is np.dtype('float32') else 'double',
            'method': 'lloyd_dense', # hardcode this???
            'cluster_count': self.n_clusters,
            'max_iteration_count': self.max_iter,
            'accuracy_threshold': self._tolerance(data, self.tol)
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
            'cluster_count': self.n_clusters,
            'random_state': self.random_state.randint(np.iinfo('i').max)
        }
    
    def _init_centroids_onedal(self, X, queue):
        module = _backend.kmeans_init.init
        policy = _get_policy(queue, X)
        params = self._get_init_onedal_params(X)
        result = module.compute(policy, params, to_table(X)) # extra conversion
        return result.centroids

    def _init_centroids(self, X, queue):
        if isinstance(self.init, str):
            centers = self._init_centroids_onedal(X, queue)
        elif _is_arraylike_not_scalar(self.init):
            centers = to_table(self.init) # should convert here?
        elif callable(self.init):
            centers = to_table(self.init(X, self.n_clusters, random_state=self.random_state))
            #some checks in stock
        return centers

    def fit(self, X, y=None, sample_weight=None, queue=None):
        module = _backend.kmeans.clustering

        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)
        self.random_state = _check_random_state(self.random_state)

        best_inertia, best_result = None, None
        for i in range(self.n_init): # getting centroids can be optimized
            centroids = self._init_centroids(X, queue)
            result = module.train(policy, params, to_table(X), centroids)#extra convert 
            inertia = result.objective_function_value
            if self.verbose:
                print(f"Iteration {i}, inertia {inertia}.")
            if best_inertia is None or inertia < best_inertia:
                best_result = result
                best_inertia = inertia

        self.n_features_in_ = X.shape[1]
        self.cluster_centers_ = from_table(best_result.model.centroids)
        self.labels_ = from_table(best_result.responses).ravel()
        self.inertia_ = best_inertia
        self.n_iter_ = best_result.iteration_count
        self._onedal_model = best_result.model
        return result

    def predict(self, X, sample_weight=None, queue=None):
        _check_is_fitted(self)

        X = self._check_test_data(X)

        module = _backend.kmeans.clustering
        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)

        model = self._onedal_model
        result = module.infer(policy, params, model, to_table(X))
        return from_table(result.responses).ravel()