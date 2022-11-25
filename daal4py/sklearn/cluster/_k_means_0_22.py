#===============================================================================
# Copyright 2014 Intel Corporation
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

try:
    from sklearn.cluster._k_means import (
        k_means, _labels_inertia, _validate_center_shape)
except ModuleNotFoundError:
    from sklearn.cluster._kmeans import (
        k_means, _labels_inertia, _validate_center_shape)

from sklearn.utils.extmath import row_norms
import warnings

from sklearn.cluster import KMeans as KMeans_original

import daal4py
from .._utils import (
    getFPType, daal_check_version, PatchingConditionsChain)
from .._device_offload import support_usm_ndarray


def _tolerance(X, rtol):
    """Compute absolute tolerance from the relative tolerance"""
    if rtol == 0.0:
        return rtol
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
        mean_var = np.mean(variances)
    else:
        mean_var = np.var(X, axis=0).mean()
    return mean_var * rtol


def _daal4py_compute_starting_centroids(
        X, X_fptype, nClusters, cluster_centers_0, random_state):

    def is_string(s, target_str):
        return isinstance(s, str) and s == target_str

    deterministic = False
    if is_string(cluster_centers_0, 'k-means++'):
        _seed = random_state.randint(np.iinfo('i').max)
        daal_engine = daal4py.engines_mt19937(
            fptype=X_fptype, method='defaultDense', seed=_seed)
        _n_local_trials = 2 + int(np.log(nClusters))
        kmeans_init = daal4py.kmeans_init(nClusters, fptype=X_fptype,
                                          nTrials=_n_local_trials,
                                          method='plusPlusDense', engine=daal_engine)
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    elif is_string(cluster_centers_0, 'random'):
        _seed = random_state.randint(np.iinfo('i').max)
        daal_engine = daal4py.engines_mt19937(
            seed=_seed, fptype=X_fptype, method='defaultDense')
        kmeans_init = daal4py.kmeans_init(
            nClusters,
            fptype=X_fptype,
            method='randomDense',
            engine=daal_engine)
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
        kmeans_init = daal4py.kmeans_init(
            nClusters, fptype=X_fptype, method='defaultDense')
        kmeans_init_res = kmeans_init.compute(X)
        centroids_ = kmeans_init_res.centroids
    else:
        raise ValueError(
            "Cluster centers should either be 'k-means++',"
            " 'random', 'deterministic' or an array")
    return deterministic, centroids_


def _daal4py_kmeans_compatibility(nClusters, maxIterations, fptype="double",
                                  method="lloydDense", accuracyThreshold=0.0,
                                  resultsToEvaluate="computeCentroids"):
    kmeans_algo = None
    if daal_check_version(((2020, 'P', 2), (2021, 'B', 107))):
        kmeans_algo = daal4py.kmeans(nClusters=nClusters,
                                     maxIterations=maxIterations,
                                     fptype=fptype,
                                     resultsToEvaluate=resultsToEvaluate,
                                     accuracyThreshold=accuracyThreshold,
                                     method=method)
    else:
        assigFlag = 'computeAssignments' in resultsToEvaluate
        kmeans_algo = daal4py.kmeans(nClusters=nClusters,
                                     maxIterations=maxIterations,
                                     fptype=fptype,
                                     assignFlag=assigFlag,
                                     accuracyThreshold=accuracyThreshold,
                                     method=method)
    return kmeans_algo


def _daal4py_k_means_predict(X, nClusters, centroids,
                             resultsToEvaluate='computeAssignments'):
    X_fptype = getFPType(X)
    kmeans_algo = _daal4py_kmeans_compatibility(
        nClusters=nClusters,
        maxIterations=0,
        fptype=X_fptype,
        resultsToEvaluate=resultsToEvaluate,
        method='defaultDense')

    res = kmeans_algo.compute(X, centroids)

    return res.assignments[:, 0], res.objectiveFunction[0, 0]


def _daal4py_k_means_fit(X, nClusters, numIterations,
                         tol, cluster_centers_0, n_init, random_state):
    if numIterations < 0:
        raise ValueError("Wrong iterations number")

    X_fptype = getFPType(X)
    abs_tol = _tolerance(X, tol)  # tol is relative tolerance

    best_inertia, best_cluster_centers = None, None
    best_n_iter = -1

    kmeans_algo = _daal4py_kmeans_compatibility(
        nClusters=nClusters,
        maxIterations=numIterations,
        accuracyThreshold=abs_tol,
        fptype=X_fptype,
        resultsToEvaluate='computeCentroids',
        method='defaultDense')

    for k in range(n_init):
        deterministic, starting_centroids_ = _daal4py_compute_starting_centroids(
            X, X_fptype, nClusters, cluster_centers_0, random_state)

        res = kmeans_algo.compute(X, starting_centroids_)

        inertia = res.objectiveFunction[0, 0]
        if best_inertia is None or inertia < best_inertia:
            best_cluster_centers = res.centroids
            if n_init > 1:
                best_cluster_centers = best_cluster_centers.copy()
            best_inertia = inertia
            best_n_iter = int(res.nIterations[0, 0])
        if deterministic and n_init != 1:
            warnings.warn(
                'Explicit initial center position passed: '
                'performing only one init in k-means instead of n_init=%d'
                % n_init, RuntimeWarning, stacklevel=2)
            break

    flag_compute = 'computeAssignments|computeExactObjectiveFunction'
    best_labels, best_inertia = _daal4py_k_means_predict(
        X, nClusters, best_cluster_centers, flag_compute)
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

    _patching_status = PatchingConditionsChain(
        "sklearn.cluster.KMeans.fit")
    _dal_ready = _patching_status.and_conditions([
        (not sp.issparse(X), "X is sparse. Sparse input is not supported."),
        (not precompute_distances, "The precomputing of distances is not supported.")
    ])

    if _dal_ready:
        X_len = _num_samples(X)
        _dal_ready = _patching_status.and_conditions([
            (self.n_clusters <= X_len,
                "The number of clusters is larger than the number of samples in X.")
        ])
        if _dal_ready and sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            _dal_ready = _patching_status.and_conditions([
                (sample_weight.shape == (X_len,),
                    "Sample weights do not have the same length as X."),
                (np.allclose(sample_weight, np.ones_like(sample_weight)),
                    "Sample weights are not ones.")
            ])

    _patching_status.write_log()
    if not _dal_ready:
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means(
                X, n_clusters=self.n_clusters, sample_weight=sample_weight,
                init=self.init, n_init=self.n_init, max_iter=self.max_iter,
                verbose=self.verbose, precompute_distances=precompute_distances,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True)
    else:
        X = check_array(
            X,
            accept_sparse='csr', dtype=[np.float64, np.float32],
            order="C" if self.copy_x else None,
            copy=self.copy_x
        )
        self.n_features_in_ = X.shape[1]
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            _daal4py_k_means_fit(
                X, self.n_clusters,
                self.max_iter,
                self.tol,
                self.init,
                self.n_init,
                random_state
            )
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
    check_is_fitted(self)

    X = self._check_test_data(X)

    _patching_status = PatchingConditionsChain(
        "sklearn.cluster.KMeans.predict")
    _dal_ready = _patching_status.and_conditions([
        (sample_weight is None, "Sample weights are not supported."),
        (hasattr(X, '__array__'), "X does not have '__array__' attribute.")
    ])

    _patching_status.write_log()
    if _dal_ready:
        return _daal4py_k_means_predict(
            X, self.n_clusters, self.cluster_centers_)[0]
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

    @support_usm_ndarray()
    def fit(self, X, y=None, sample_weight=None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return _fit(self, X, y=y, sample_weight=sample_weight)

    @support_usm_ndarray()
    def predict(self, X, sample_weight=None):
        """
        Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return _predict(self, X, sample_weight=sample_weight)

    @support_usm_ndarray()
    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return super().fit_predict(X, y, sample_weight)
