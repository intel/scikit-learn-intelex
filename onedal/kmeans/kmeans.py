# ===============================================================================
# Copyright 2023 Intel Corporation
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
# ===============================================================================

import warnings

import numpy as np

from onedal import _backend

from abc import ABC, abstractmethod

from daal4py.sklearn._utils import get_dtype
from ..datatypes import _convert_to_supported

from ..kmeans_init import KMeansInit

from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted
from ..datatypes._data_conversion import from_table, to_table

from sklearn.utils import check_random_state
from sklearn.utils.validation import _is_arraylike_not_scalar

from sklearn.base import (BaseEstimator, ClusterMixin, TransformerMixin)

from sklearn.cluster._k_means_common import _inertia_dense

class _BaseKMeans(ClusterMixin, BaseEstimator, ABC):
    def __init__(self, n_clusters, *, init, n_init, max_iter, tol, verbose, random_state, n_local_trials = None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self._cluster_centers_ = None
        self.n_local_trials = n_local_trials
        self.random_state = check_random_state(random_state)

        default_n_init = 10
        self._n_init = self.n_init
        if self._n_init == "warn":
            warnings.warn(
                "The default value of `n_init` will change from "
                f"{default_n_init} to 'auto' in 1.4. Set the value of `n_init`"
                " explicitly to suppress the warning",
                FutureWarning,
            )
            self._n_init = default_n_init
        if self._n_init == "auto":
            if self.init == "k-means++":
                self._n_init = 1
            else:
                self._n_init = default_n_init

    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _get_onedal_params(self, dtype=np.float32):
        if self.algorithm != "lloyd":
            raise ValueError("Only \"lloyd\" algorithm is supported")
        return {
            'fptype': 'float' if dtype == np.float32 else 'double',
            'method': 'by_default', 'seed': -1,
            'max_iteration_count': self.max_iter,
            'cluster_count': self.n_clusters,
            'accuracy_threshold': self.tol,
        }

    def _get_params_and_input(self, X, policy):
        X_loc = np.asarray(X)
        types = [np.float32, np.float64]
        if get_dtype(X_loc) not in types:
            X_loc = X_loc.astype(np.float64)

        X_loc = _convert_to_supported(policy, X_loc)

        dtype = get_dtype(X_loc)
        params = self._get_onedal_params(dtype)
        return (params, to_table(X_loc), dtype)

    def _init_centroids_raw(self, X_table, init, random_seed, policy, dtype = np.float32):
        if isinstance(init, str) and init == "k-means++":
            alg = KMeansInit(cluster_count = self.n_clusters, seed = random_seed, algorithm = "plus_plus_dense")
            centers_table = alg.compute_raw(X_table, policy, dtype)
        elif isinstance(init, str) and init == "random":
            alg = KMeansInit(cluster_count = self.n_clusters, seed = random_seed, algorithm = "random_dense")
            centers_table = alg.compute_raw(X_table, policy, dtype)
        elif _is_arraylike_not_scalar(init):
            centers = np.asarray(init)
            assert centers.shape[0] == self.n_clusters
            assert centers.shape[1] == X_table.column_count
            centers = _convert_to_supported(policy, init)
            centers_table = to_table(centers)
        else:
            raise TypeError("Unsupported type of the `init` value")

        return centers_table

    def _fit_backend(self, X_table, centroids_table, module, policy, dtype = np.float32):
        params = self._get_onedal_params(dtype)

        # TODO: check all features for having correct type
        assert _backend.get_table_column_type(centroids_table, 0) == dtype

        result = module.train(policy, params, X_table, centroids_table)

        return ( result.responses, result.objective_function_value, \
                        result.model, result.iteration_count )

    def _fit(self, X, module, queue = None):
        policy = self._get_policy(queue, X)
        params, X_table, dtype = self._get_params_and_input(X, policy)

        best_model, best_n_iter = None, None
        best_inertia, best_labels = None, None

        def is_best_inertia(inertia, labels):
            if best_inertia is None:
                return True
            else:
                return inertia < best_inertia

        random_state = check_random_state(self.random_state)

        for i in range(self._n_init):
            random_seed = random_state.tomaxint()
            centroids_table = self._init_centroids_raw(
                X_table, self.init, random_seed, policy, dtype
            )

            if self.verbose:
                print("Initialization complete")

            labels, inertia, model, n_iter = self._fit_backend(
                X_table, centroids_table, module, policy, dtype
            )

            if self.verbose:
                print("KMeans iteration completed with "
                      "inertia {}.".format(inertia)
                )

            if is_best_inertia(inertia, labels):
                best_model, best_n_iter = model, n_iter
                best_inertia, best_labels = inertia, labels

        labels = from_table(best_labels)
        distinct_clusters = len(np.unique(labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.labels_ = labels
        self.model_ = best_model
        self.n_iter_ = best_n_iter
        self.inertia_ = best_inertia

        return self

    def get_cluster_centers(self):
        if not hasattr(self, "_cluster_centers_"):
            return self._cluster_centers_
            if hasattr(self, "model_"):
                centroids = self.model_.centroids
                self._cluster_centers_ = to_table(centroids)
            else:
                raise NameError("This model have not been trained")
        return self._cluster_centers_

    def set_cluster_centers(self, cluster_centers):
        self._cluster_centers_ = np.asarray(cluster_centers)

        self.n_iter_ = 0
        self.inertia_ = 0

        self.model_ = module.model()
        self.model_.centroids = to_table(self._cluster_centers_)

        return self

    cluster_centers_ = property(get_cluster_centers, set_cluster_centers)

    def _predict(self, X, module, queue = None):
        policy = self._get_policy(queue, X)

        params, X_table, dtype = self._get_params_and_input(X, policy)

        return module.infer(policy, params, self.model_, X_table)


class KMeans(_BaseKMeans):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

        self.copy_x = copy_x
        self.algorithm = algorithm

    def fit(self, X, queue = None):
        return super()._fit(X, _backend.kmeans.clustering, queue)

    def predict(self, X, queue = None):
        return super()._predict(X, _backend.kmeans.clustering, queue)


def k_means(
    X,
    n_clusters,
    *,
    init="k-means++",
    n_init="warn",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
    queue = None
):
    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, queue)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
