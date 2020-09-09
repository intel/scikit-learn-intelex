# *******************************************************************************
# Copyright 2014-2020 Intel Corporation
# All Rights Reserved.
#
# This software is licensed under the Apache License, Version 2.0 (the
# "License"), the following terms apply:
#
# You may not use this file except in compliance with the License.  You may
# obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# daal4py KNN scikit-learn-compatible estimator class

import numpy as np
import numbers
import daal4py as d4p
from .._utils import getFPType
from functools import partial
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from distutils.version import LooseVersion
from scipy import stats
from sklearn.utils.extmath import weighted_mode
from sklearn.utils.validation import _num_samples
from sklearn.neighbors._base import \
    _check_weights, _get_weights, \
    NeighborsBase, SupervisedIntegerMixin
from sklearn.utils.validation import _deprecate_positional_args
from scipy.sparse import csr_matrix, issparse
import joblib
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils import gen_even_slices
from sklearn.neighbors._base import \
    _check_precomputed, _tree_query_parallel_helper, _kneighbors_from_graph


class KNeighborsMixin:
    """Mixin for k-neighbors searches"""

    def _kneighbors_reduce_func(self, dist, start,
                                n_neighbors, return_distance):
        """Reduce a chunk of distances to the nearest neighbors

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : array of shape (n_samples_chunk, n_samples)
        start : int
            The index in X which the first row of dist corresponds to.
        n_neighbors : int
        return_distance : bool

        Returns
        -------
        dist : array of shape (n_samples_chunk, n_neighbors), optional
            Returned only if return_distance
        neigh : array of shape (n_samples_chunk, n_neighbors)
        """
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[
            sample_range, np.argsort(dist[sample_range, neigh_ind])]
        if return_distance:
            if self.effective_metric_ == 'euclidean':
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = dist[sample_range, neigh_ind], neigh_ind
        else:
            result = neigh_ind
        return result

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        neigh_dist : array, shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True

        neigh_ind : array, shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construct a NearestNeighbors
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]

        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples)
        NearestNeighbors(n_neighbors=1)
        >>> print(neigh.kneighbors([[1., 1., 1.]]))
        (array([[0.5]]), array([[2]]))

        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:

        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False)
        array([[1],
               [2]]...)

        """
        check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(
                "Expected n_neighbors > 0. Got %d" %
                n_neighbors
            )
        else:
            if not isinstance(n_neighbors, numbers.Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(n_neighbors))

        if X is not None:
            query_is_train = False
            if self.effective_metric_ == 'precomputed':
                X = _check_precomputed(X)
            else:
                X = check_array(X, accept_sparse='csr')
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        n_samples_fit = self.n_samples_fit_
        if n_neighbors > n_samples_fit:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (n_samples_fit, n_neighbors)
            )

        n_jobs = effective_n_jobs(self.n_jobs)
        chunked_results = None

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        if self._fit_method in ['brute', 'kd_tree'] \
        and (self.effective_metric_ == 'minkowski' and self.p == 2 or self.effective_metric_ == 'euclidean') \
        and fptype is not None:

            if self._fit_method == 'brute':
                knn_classification_training = d4p.bf_knn_classification_training
                knn_classification_prediction = d4p.bf_knn_classification_prediction
                # Brute force method always computes in doubles due to precision need
                compute_fptype = 'double'
            else:
                knn_classification_training = d4p.kdtree_knn_classification_training
                knn_classification_prediction = d4p.kdtree_knn_classification_prediction
                compute_fptype = fptype

            alg_params = {
                'fptype': compute_fptype,
                'method': 'defaultDense',
                'k': n_neighbors,
                'resultsToCompute': 'computeIndicesOfNeightbors|computeDistances',
                'resultsToEvaluate': 'none'
            }

            training_alg = knn_classification_training(**alg_params)

            fit_X = d4p.get_data(self._fit_X)
            training_result = training_alg.compute(fit_X)

            prediction_alg = knn_classification_prediction(**alg_params)

            X = d4p.get_data(X)
            prediction_result = prediction_alg.compute(X, training_result.model)

            if return_distance:
                results = prediction_result.distances.astype(fptype), prediction_result.indices
            else:
                results = prediction_result.indices

        elif (self._fit_method == 'brute' and
                self.effective_metric_ == 'precomputed' and issparse(X)):
            results = _kneighbors_from_graph(
                X, n_neighbors=n_neighbors,
                return_distance=return_distance)

        elif self._fit_method == 'brute':
            reduce_func = partial(self._kneighbors_reduce_func,
                                  n_neighbors=n_neighbors,
                                  return_distance=return_distance)

            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == 'euclidean':
                kwds = {'squared': True}
            else:
                kwds = self.effective_metric_params_

            chunked_results = list(pairwise_distances_chunked(
                X, self._fit_X, reduce_func=reduce_func,
                metric=self.effective_metric_, n_jobs=n_jobs,
                **kwds))

        elif self._fit_method in ['ball_tree', 'kd_tree']:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'" % self._fit_method)
            old_joblib = (
                    LooseVersion(joblib.__version__) < LooseVersion('0.12'))
            if old_joblib:
                # Deal with change of API in joblib
                check_pickle = False if old_joblib else None
                delayed_query = delayed(_tree_query_parallel_helper,
                                        check_pickle=check_pickle)
                parallel_kwargs = {"backend": "threading"}
            else:
                delayed_query = delayed(_tree_query_parallel_helper)
                parallel_kwargs = {"prefer": "threads"}
            chunked_results = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(
                    self._tree, X[s], n_neighbors, return_distance)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
        else:
            raise ValueError("internal: _fit_method not recognized")

        if chunked_results is not None:
            if return_distance:
                neigh_dist, neigh_ind = zip(*chunked_results)
                results = np.vstack(neigh_dist), np.vstack(neigh_ind)
            else:
                results = np.vstack(chunked_results)

        if not query_is_train:
            return results
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                neigh_dist, neigh_ind = results
            else:
                neigh_ind = results

            n_queries, _ = X.shape
            sample_range = np.arange(n_queries)[:, None]
            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False
            neigh_ind = np.reshape(
                neigh_ind[sample_mask], (n_queries, n_neighbors - 1))

            if return_distance:
                neigh_dist = np.reshape(
                    neigh_dist[sample_mask], (n_queries, n_neighbors - 1))
                return neigh_dist, neigh_ind
            return neigh_ind

    def kneighbors_graph(self, X=None, n_neighbors=None,
                         mode='connectivity'):
        """Computes the (weighted) graph of k-Neighbors for points in X

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int
            Number of neighbors for each sample.
            (default is value passed to the constructor).

        mode : {'connectivity', 'distance'}, optional
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are Euclidean distance between points.

        Returns
        -------
        A : sparse graph in CSR format, shape = [n_queries, n_samples_fit]
            n_samples_fit is the number of samples in the fitted data
            A[i, j] is assigned the weight of edge that connects i to j.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=2)
        >>> neigh.fit(X)
        NearestNeighbors(n_neighbors=2)
        >>> A = neigh.kneighbors_graph(X)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 0., 1.]])

        See also
        --------
        NearestNeighbors.radius_neighbors_graph
        """
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # check the input only in self.kneighbors

        # construct CSR matrix representation of the k-NN graph
        if mode == 'connectivity':
            A_ind = self.kneighbors(X, n_neighbors, return_distance=False)
            n_queries = A_ind.shape[0]
            A_data = np.ones(n_queries * n_neighbors)

        elif mode == 'distance':
            A_data, A_ind = self.kneighbors(
                X, n_neighbors, return_distance=True)
            A_data = np.ravel(A_data)

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity" '
                'or "distance" but got "%s" instead' % mode)

        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_nonzero = n_queries * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

        kneighbors_graph = csr_matrix((A_data, A_ind.ravel(), A_indptr),
                                      shape=(n_queries, n_samples_fit))

        return kneighbors_graph


class KNeighborsClassifier(NeighborsBase, KNeighborsMixin,
                           SupervisedIntegerMixin, ClassifierMixin):

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

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr')

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        if self.weights in ['uniform', 'distance'] and self.algorithm in ['brute', 'kd_tree'] \
        and (self.metric == 'minkowski' and self.p == 2 or self.metric == 'euclidean') \
        and self._y.ndim == 1 and fptype is not None:

            n_classes = len(self.classes_)

            if self.algorithm == 'brute':
                knn_classification_training = d4p.bf_knn_classification_training
                knn_classification_prediction = d4p.bf_knn_classification_prediction
                # Brute force method always computes in doubles due to precision need
                compute_fptype = 'double'
            else:
                knn_classification_training = d4p.kdtree_knn_classification_training
                knn_classification_prediction = d4p.kdtree_knn_classification_prediction
                compute_fptype = fptype

            alg_params = {
                'fptype': compute_fptype,
                'method': 'defaultDense',
                'k': self.n_neighbors,
                'nClasses': n_classes,
                'voteWeights': 'voteUniform' if self.weights == 'uniform' else 'voteDistance',
                'resultsToEvaluate': 'computeClassLabels'
            }

            training_alg = knn_classification_training(**alg_params)

            fit_X = d4p.get_data(self._fit_X)
            _y = d4p.get_data(self._y)
            _y = _y.reshape(_y.shape[0], 1)
            training_result = training_alg.compute(fit_X, _y)

            prediction_alg = knn_classification_prediction(**alg_params)

            X = d4p.get_data(X)
            prediction_result = prediction_alg.compute(X, training_result.model)
            return prediction_result.prediction.ravel().astype(self._y.dtype)

        else:
            neigh_dist, neigh_ind = self.kneighbors(X)
            classes_ = self.classes_
            _y = self._y
            if not self.outputs_2d_:
                _y = self._y.reshape((-1, 1))
                classes_ = [self.classes_]

            n_outputs = len(classes_)
            n_queries = _num_samples(X)
            weights = _get_weights(neigh_dist, self.weights)

            y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
            for k, classes_k in enumerate(classes_):
                if weights is None:
                    mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
                else:
                    mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

                mode = np.asarray(mode.ravel(), dtype=np.intp)
                y_pred[:, k] = classes_k.take(mode)

            if not self.outputs_2d_:
                y_pred = y_pred.ravel()

            return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X)

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(X.shape[0])
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities
