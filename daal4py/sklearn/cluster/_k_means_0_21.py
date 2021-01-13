#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

import numpy as np
from scipy import sparse as sp
from sklearn.utils import (check_random_state, check_array)
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import (check_is_fitted, _num_samples)
from sklearn.cluster.k_means_ import (k_means, _labels_inertia, _validate_center_shape)

from sklearn.utils.extmath import row_norms
import warnings

from sklearn.cluster import KMeans as KMeans_original

import daal4py
from .._utils import getFPType, get_patch_message, daal_check_version
import logging

def _daal_mean_var(X):
    fpt = getFPType(X)
    try:
        alg = daal4py.low_order_moments(fptype=fpt, method='defaultDense', estimatesToCompute='estimatesAll')
    except AttributeError:
        return np.var(X, axis=0).mean()
    ssc = alg.compute(X).sumSquaresCentered
    ssc = ssc.reshape((-1,1))
    alg = daal4py.low_order_moments(fptype=fpt, method='defaultDense', estimatesToCompute='estimatesAll')
    ssc_total_res = alg.compute(ssc)
    mean_var = ssc_total_res.sum / X.size
    return mean_var[0, 0]


def _tolerance(X, rtol):
    """Compute absolute tolerance from the relative tolerance"""
    if rtol == 0.0:
        return rtol
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
        mean_var = np.mean(variances)
    else:
        mean_var = _daal_mean_var(X)
    return mean_var * rtol

def _daal4py_compute_starting_centroids(X, X_fptype, nClusters, cluster_centers_0, random_state):

    def is_string(s, target_str):
        return isinstance(s, str) and s == target_str

    deterministic = False
    if is_string(cluster_centers_0, 'k-means++'):
        _seed = random_state.randint(np.iinfo('i').max)
        daal_engine = daal4py.engines_mt19937(fptype=X_fptype, method='defaultDense', seed=_seed)
        _n_local_trials = 2 + int(np.log(nClusters))
        kmeans_init = daal4py.kmeans_init(nClusters, fptype=X_fptype,
                                          nTrials=_n_local_trials, method='plusPlusDense', engine=daal_engine)
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    elif is_string(cluster_centers_0, 'random'):
        _seed = random_state.randint(np.iinfo('i').max)
        daal_engine = daal4py.engines_mt19937(seed=_seed, fptype=X_fptype, method='defaultDense')
        kmeans_init = daal4py.kmeans_init(nClusters, fptype=X_fptype, method='randomDense', engine=daal_engine)
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    elif hasattr(cluster_centers_0, '__array__'):
        deterministic = True
        cc_arr = np.ascontiguousarray(cluster_centers_0, dtype=X.dtype)
        _validate_center_shape(X, nClusters, cc_arr)
        centroids_ = cc_arr
    elif callable(cluster_centers_0):
        cc_arr = cluster_centers_0(X, nClusters, random_state)
        cc_arr = np.ascontiguousarray(cc_arr, dtype=X.dtype)
        _validate_center_shape(X, nClusters, cc_arr)
        centroids_ = cc_arr
    elif is_string(cluster_centers_0, 'deterministic'):
        deterministic = True
        kmeans_init = daal4py.kmeans_init(nClusters, fptype=X_fptype, method='defaultDense')
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    else:
        raise ValueError("Cluster centers should either be 'k-means++', 'random', 'deterministic' or an array")
    return deterministic, centroids_

def _daal4py_kmeans_compatibility(nClusters, maxIterations, fptype = "double",
    method = "lloydDense", accuracyThreshold = 0.0, resultsToEvaluate = "computeCentroids"):
    kmeans_algo = None
    if daal_check_version(((2020,'P', 2), (2021,'B',107))):
        kmeans_algo = daal4py.kmeans(nClusters = nClusters,
            maxIterations= maxIterations,
            fptype = fptype,
            resultsToEvaluate = resultsToEvaluate,
            accuracyThreshold=accuracyThreshold,
            method = method)
    else:
        assigFlag = 'computeAssignments' in resultsToEvaluate
        kmeans_algo = daal4py.kmeans(nClusters = nClusters,
            maxIterations= maxIterations,
            fptype = fptype,
            assignFlag = assigFlag,
            accuracyThreshold=accuracyThreshold,
            method = method)
    return kmeans_algo


def _daal4py_k_means_predict(X, nClusters, centroids, resultsToEvaluate = 'computeAssignments'):
    X_fptype = getFPType(X)
    kmeans_algo = _daal4py_kmeans_compatibility(
        nClusters = nClusters,
        maxIterations = 0,
        fptype = X_fptype,
        resultsToEvaluate = resultsToEvaluate,
        method = 'defaultDense')

    res = kmeans_algo.compute(X, centroids)

    return res.assignments[:,0], res.objectiveFunction[0,0]


def _daal4py_k_means_fit(X, nClusters, numIterations, tol, cluster_centers_0, n_init, random_state):
    if numIterations < 0:
        raise ValueError("Wrong iterations number")

    X_fptype = getFPType(X)
    abs_tol = _tolerance(X, tol) # tol is relative tolerance

    best_labels, best_inertia, best_cluster_centers = None, None, None
    best_n_iter = -1

    kmeans_algo = _daal4py_kmeans_compatibility(
        nClusters = nClusters,
        maxIterations = numIterations,
        accuracyThreshold = abs_tol,
        fptype = X_fptype,
        resultsToEvaluate = 'computeCentroids',
        method = 'defaultDense')

    for k in range(n_init):
        deterministic, starting_centroids_ = _daal4py_compute_starting_centroids(
            X, X_fptype, nClusters, cluster_centers_0, random_state)

        res = kmeans_algo.compute(X, starting_centroids_)

        inertia = res.objectiveFunction[0,0]
        if best_inertia is None or inertia < best_inertia:
            best_cluster_centers = res.centroids
            if n_init > 1:
                best_cluster_centers = best_cluster_centers.copy()
            best_inertia = inertia

        if deterministic and n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            break

    flag_compute = 'computeAssignments|computeExactObjectiveFunction'
    best_labels, best_inertia = _daal4py_k_means_predict(X, nClusters, best_cluster_centers, flag_compute)
    return best_cluster_centers, best_labels, best_inertia, best_n_iter


def _fit(self, X, y=None, sample_weight=None):
    """Compute k-means clustering.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.

    y : Ignored
        not used, present here for API consistency by convention.

    sample_weight : array-like, shape (n_samples,), optional
        The weights for each observation in X. If None, all observations
        are assigned equal weight (default: None)

    """
    if self.n_init <= 0:
        raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % self.n_init)

    random_state = check_random_state(self.random_state)

    if self.max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % self.max_iter)

    if self.precompute_distances == 'auto':
        precompute_distances = False
    elif isinstance(self.precompute_distances, bool):
        precompute_distances = self.precompute_distances
    else:
        raise ValueError("precompute_distances should be 'auto' or True/False"
                         ", but a value of %r was passed" %
                         self.precompute_distances)

    # avoid forcing order when copy_x=False
    order = "C" if self.copy_x else None
    X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32],
                    order=order, copy=self.copy_x)

    daal_ready = not sp.issparse(X) and not precompute_distances
    daal_ready = daal_ready and hasattr(X, '__array__')

    if daal_ready:
        X_len = _num_samples(X)
        daal_ready = (self.n_clusters <= X_len)
        if daal_ready and sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            daal_ready = (sample_weight.shape[0] == X_len) and (
                         np.allclose(sample_weight, np.ones_like(sample_weight)))

    if not daal_ready:
        logging.info("sklearn.cluster.KMeans.fit: " + get_patch_message("sklearn"))
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means(
                X, n_clusters=self.n_clusters, sample_weight=sample_weight, init=self.init,
                n_init=self.n_init, max_iter=self.max_iter, verbose=self.verbose,
                precompute_distances=precompute_distances,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True)
    else:
        logging.info("sklearn.cluster.KMeans.fit: " + get_patch_message("daal"))
        X = check_array(X, dtype=[np.float64, np.float32])
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            _daal4py_k_means_fit(
                X, self.n_clusters, self.max_iter, self.tol, self.init, self.n_init,
                random_state)
    return self


def _predict(self, X, sample_weight=None):
    """Predict the closest cluster each sample in X belongs to.

    In the vector quantization literature, `cluster_centers_` is called
    the code book and each value returned by `predict` is the index of
    the closest code in the code book.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
       New data to predict.

    sample_weight : array-like, shape (n_samples,), optional
        The weights for each observation in X. If None, all observations
        are assigned equal weight (default: None)

    Returns
    -------
    labels : array, shape [n_samples,]
        Index of the cluster each sample belongs to.
    """
    check_is_fitted(self, 'cluster_centers_')

    X = self._check_test_data(X)

    daal_ready = sample_weight is None and hasattr(X, '__array__') # or sp.isspmatrix_csr(X)

    if daal_ready:
        logging.info("sklearn.cluster.KMeans.predict: " + get_patch_message("daal"))
        return _daal4py_k_means_predict(X, self.n_clusters, self.cluster_centers_)[0]
    logging.info("sklearn.cluster.KMeans.predict: " + get_patch_message("sklearn"))
    x_squared_norms = row_norms(X, squared=True)
    return _labels_inertia(X, sample_weight, x_squared_norms,
                           self.cluster_centers_)[0]


class KMeans(KMeans_original):
    __doc__ = KMeans_original.__doc__

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):

        super(KMeans, self).__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            tol=tol, precompute_distances=precompute_distances,
            n_init=n_init, verbose=verbose, random_state=random_state,
            copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)

    def fit(self, X, y=None, sample_weight=None):
        return _fit(self, X, y=y, sample_weight=sample_weight)

    def predict(self, X, sample_weight=None):
        return _predict(self, X, sample_weight=sample_weight)
