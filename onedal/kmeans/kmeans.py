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
from scipy import sparse as sp

from ..common._estimator_checks import _check_is_fitted

import warnings # move warnings to sklearnex?????
from sklearn.utils.sparsefuncs import mean_variance_axis # tmp
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads #move???
from sklearn.exceptions import ConvergenceWarning # move???

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
        if sp.issparse(X):
            variances = mean_variance_axis(X, axis=0)[1]
            mean_var = np.mean(variances)
        else:
            mean_var = np.var(X, axis=0).mean()
        return mean_var * rtol

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )
    
    def _check_params(self, X): #refactor this method
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        self._tol = self._tolerance(X, self.tol)

        if not (
            _is_arraylike_not_scalar(self.init)
            or callable(self.init)
            or (isinstance(self.init, str) and self.init in ["k-means++", "random"])
        ):
            raise ValueError(
                "init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead."
            )

        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1

#split??

        algorithm = self.algorithm
        if algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn("algorithm='elkan' doesn't make sense for a single "
                        "cluster. Using 'full' instead.", RuntimeWarning)
            algorithm = "full"

        if algorithm == "auto":
            algorithm = "full" if self.n_clusters == 1 else "elkan"

        if algorithm not in ["full", "elkan"]:
            raise ValueError("Algorithm must be 'auto', 'full' or 'elkan', got"
                            " {}".format(str(algorithm)))

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
            init = self.init
            init = _check_array(init, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, init)
            centers = to_table(init) # should convert here?
        elif callable(self.init):
            init = self.init(X, self.n_clusters, random_state=self.random_state)
            init = _check_array(init, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, init)
            centers = to_table(init)
        if self.verbose:
            print("Initialization complete")
        return centers

    def fit(self, X, y=None, sample_weight=None, queue=None):
        X = _check_array(
            X,
            accept_sparse='csr',
            dtype=[np.float64, np.float32],
            accept_large_sparse=False
        )
        self._check_params(X)

        self._n_threads = None
        if hasattr(self, 'n_jobs'):
            if self.n_jobs != 'deprecated':
            ## n_jobs handling
                if sklearn_check_version('0.24'):
                    warnings.warn("'n_jobs' was deprecated in version 0.23 and will be"
                                " removed in 1.0 (renaming of 0.25).", FutureWarning)
                elif sklearn_check_version('0.23'):
                    warnings.warn("'n_jobs' was deprecated in version 0.23 and will be"
                                " removed in 0.25.", FutureWarning)
                self._n_threads = self.n_jobs
        self._n_threads = _openmp_effective_n_threads(self._n_threads)

        module = _backend.kmeans.clustering
        policy = _get_policy(queue, X)
        params = self._get_onedal_params(X)
        self.random_state = _check_random_state(self.random_state)

        best_inertia, best_result = None, None
        for i in range(self.n_init): # getting centroids can be optimized
            centroids = self._init_centroids(X, queue)
            result = module.train(policy, params, to_table(X), centroids) #extra convert 
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
