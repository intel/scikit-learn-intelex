#
# *******************************************************************************
# Copyright 2020 Intel Corporation
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
# ******************************************************************************/

# daal4py KNN scikit-learn-compatible base classes

import numpy as np
import numbers
import daal4py as d4p
from scipy import sparse as sp
from .._utils import getFPType, daal_check_version, method_uses_sklearn, method_uses_daal
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors._base import KNeighborsMixin as BaseKNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as BaseNeighborsBase
from joblib import effective_n_jobs
from sklearn.neighbors._base import _check_precomputed
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import is_classifier
import logging


def training_algorithm(method, fptype, params):
    if method == 'brute':
        train_alg = d4p.bf_knn_classification_training
        # Brute force method always computes in doubles due to precision need
        compute_fptype = 'double'
    else:
        train_alg = d4p.kdtree_knn_classification_training
        compute_fptype = fptype

    params['fptype'] = compute_fptype

    return train_alg(**params)


def prediction_algorithm(method, fptype, params):
    if method == 'brute':
        predict_alg = d4p.bf_knn_classification_prediction
        # Brute force method always computes in doubles due to precision need
        compute_fptype = 'double'
    else:
        predict_alg = d4p.kdtree_knn_classification_prediction
        compute_fptype = fptype

    params['fptype'] = compute_fptype

    return predict_alg(**params)


def parse_auto_method(clf, method, n_samples, n_features):
    result_method = method

    if (method == 'auto'):
        if clf.metric == 'precomputed' or n_features > 11 or \
           (clf.n_neighbors is not None and clf.n_neighbors >= clf.n_neighbors // 2):
            result_method = 'brute'
        else:
            if clf.effective_metric_ in KDTree.valid_metrics:
                result_method = 'kd_tree'
            else:
                result_method = 'brute'

    return result_method


def daal4py_fit(estimator, X, fptype):
    estimator.n_samples_fit_ = X.shape[0]
    estimator.n_features_in_ = X.shape[1]
    estimator._fit_X = X
    estimator._fit_method = estimator.algorithm
    estimator.effective_metric_ = 'euclidean'

    params = {
        'method': 'defaultDense',
        'k': estimator.n_neighbors,
        'nClasses': len(estimator.classes_),
        'voteWeights': 'voteUniform' if estimator.weights == 'uniform' else 'voteDistance',
        'resultsToEvaluate': 'computeClassLabels',
        'resultsToCompute': ''
    }

    method = parse_auto_method(estimator, estimator.algorithm, estimator.n_samples_fit_, estimator.n_features_in_)
    train_alg = training_algorithm(method, fptype, params)
    estimator.daal_model_ = train_alg.compute(X, estimator._y.reshape(-1, 1)).model


def daal4py_kneighbors(estimator, X=None, n_neighbors=None, return_distance=True):
    n_features = getattr(estimator, 'n_features_in_', None)
    shape = getattr(X, 'shape', None)
    if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        raise ValueError('Input data shape {} is inconsistent with the trained model'.format(X.shape))

    check_is_fitted(estimator)

    if n_neighbors is None:
        n_neighbors = estimator.n_neighbors
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
        X = check_array(X, accept_sparse='csr')
    else:
        query_is_train = True
        X = estimator._fit_X
        # Include an extra neighbor to account for the sample itself being
        # returned, which is removed later
        n_neighbors += 1

    n_samples_fit = estimator.n_samples_fit_
    if n_neighbors > n_samples_fit:
        raise ValueError(
            "Expected n_neighbors <= n_samples, "
            " but n_samples = %d, n_neighbors = %d" %
            (n_samples_fit, n_neighbors)
        )

    chunked_results = None

    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    params = {
        'method': 'defaultDense',
        'k': n_neighbors,
        'resultsToCompute': 'computeIndicesOfNeighbors',
        'resultsToEvaluate': 'none'
    }
    if return_distance:
        params['resultsToCompute'] += '|computeDistances'

    method = parse_auto_method(estimator, estimator._fit_method, estimator.n_samples_fit_, n_features)

    predict_alg = prediction_algorithm(method, fptype, params)
    prediction_result = predict_alg.compute(X, estimator.daal_model_)

    if return_distance:
        results = prediction_result.distances.astype(fptype), prediction_result.indices.astype(int)
    else:
        results = prediction_result.indices.astype(int)

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


class NeighborsBase(BaseNeighborsBase):
    def _fit(self, X, y=None):
        X_incorrect_type = isinstance(X, (KDTree, BallTree, NeighborsBase, BaseNeighborsBase))

        if not X_incorrect_type:
            X, y = self._validate_data(X, y, accept_sparse="csr", multi_output=True)
            single_output = False if y.ndim > 1 and y.shape[1] > 1 else True

        if is_classifier(self):
            if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                self.outputs_2d_ = False
                y = y.reshape((-1, 1))
            else:
                self.outputs_2d_ = True

            check_classification_targets(y)
            self.classes_ = []
            self._y = np.empty(y.shape, dtype=int)
            for k in range(self._y.shape[1]):
                classes, self._y[:, k] = np.unique(
                    y[:, k], return_inverse=True)
                self.classes_.append(classes)

            if not self.outputs_2d_:
                self.classes_ = self.classes_[0]
                self._y = self._y.ravel()

            n_classes = len(self.classes_)
            if n_classes < 2:
                raise ValueError("Training data only contain information about one class.")
        else:
            self._y = y

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        if daal_check_version((2020, 3)) and not X_incorrect_type \
        and self.weights in ['uniform', 'distance'] and self.algorithm in ['brute', 'kd_tree', 'auto'] \
        and (self.metric == 'minkowski' and self.p == 2 or self.metric == 'euclidean') \
        and single_output and fptype is not None and not sp.issparse(X):
            logging.info("sklearn.neighbors.NeighborsBase._fit: " + method_uses_daal)
            daal4py_fit(self, X, fptype)
            result = self
        else:
            logging.info("sklearn.neighbors.NeighborsBase._fit: " + method_uses_sklearn)
            result = super(NeighborsBase, self)._fit(X)

        return result


class KNeighborsMixin(BaseKNeighborsMixin):
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        x = self._fit_X if X is None else X
        try:
            fptype = getFPType(x)
        except ValueError:
            fptype = None

        if daal_check_version((2020, 3)) and hasattr(self, 'daal_model_') \
        and self._fit_method in ['brute', 'kd_tree', 'auto'] \
        and (self.effective_metric_ == 'minkowski' and self.p == 2 or self.effective_metric_ == 'euclidean') \
        and fptype is not None and not sp.issparse(X):
            logging.info("sklearn.neighbors.KNeighborsMixin.kneighbors: " + method_uses_daal)
            result = daal4py_kneighbors(self, X, n_neighbors, return_distance)
        else:
            logging.info("sklearn.neighbors.KNeighborsMixin.kneighbors: " + method_uses_sklearn)
            if hasattr(self, 'daal_model_'):
                BaseNeighborsBase._fit(self, self._fit_X, self._y)
            result = super(KNeighborsMixin, self).kneighbors(X, n_neighbors, return_distance)

        return result
