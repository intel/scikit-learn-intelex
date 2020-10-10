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

# daal4py KNN scikit-learn-compatible estimator classes

import numpy as np
import numbers
import daal4py as d4p
from scipy import sparse as sp
from .._utils import getFPType, daal_check_version, method_uses_sklearn, method_uses_daal, make2d
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors._base import KNeighborsMixin as BaseKNeighborsMixin
from sklearn.neighbors._classification import KNeighborsClassifier as BaseKNeighborsClassifier
from joblib import effective_n_jobs
from sklearn.neighbors._base import _check_precomputed, NeighborsBase
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._kd_tree import KDTree
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder
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


class KNeighborsMixin(BaseKNeighborsMixin):
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        n_features = getattr(self, 'n_features_in_', None)
        shape = getattr(X, 'shape', None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError('Input data shape {} is inconsistent with the trained model'.format(X.shape))

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

        chunked_results = None

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        fit_X_correct_type = isinstance(self._fit_X, np.ndarray)

        if daal_check_version(((2020,'P', 3),(2021,'B', 110))) and fit_X_correct_type and self._fit_method in ['brute', 'kd_tree', 'auto'] \
        and (self.effective_metric_ == 'minkowski' and self.p == 2 or self.effective_metric_ == 'euclidean') \
        and fptype is not None and not sp.issparse(X):
            logging.info("sklearn.neighbors.KNeighborsMixin.kneighbors: " + method_uses_daal)

            params = {
                'method': 'defaultDense',
                'k': n_neighbors,
                'resultsToCompute': 'computeIndicesOfNeighbors',
                'resultsToEvaluate': 'none'
            }
            if return_distance:
                params['resultsToCompute'] += '|computeDistances'

            method = parse_auto_method(self, self._fit_method, self.n_samples_fit_, n_features)

            fit_X = d4p.get_data(self._fit_X)
            train_alg = training_algorithm(method, fptype, params)
            training_result = train_alg.compute(fit_X)

            X = d4p.get_data(X)
            predict_alg = prediction_algorithm(method, fptype, params)
            prediction_result = predict_alg.compute(X, training_result.model)

            if return_distance:
                results = prediction_result.distances.astype(fptype), prediction_result.indices.astype(int)
            else:
                results = prediction_result.indices.astype(int)
        else:
            logging.info("sklearn.neighbors.KNeighborsMixin.kneighbors: " + method_uses_sklearn)
            return super(KNeighborsMixin, self).kneighbors(X, n_neighbors, return_distance)

        if chunked_results is not None:
            if return_distance:
                neigh_dist, neigh_ind = zip(*chunked_results)
                results = np.vstack(neigh_dist), np.vstack(neigh_ind)
            else:
                results = np.vstack(chunked_results)

        if not query_is_train:
            return results
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


class KNeighborsClassifier(BaseKNeighborsClassifier, KNeighborsMixin):
    def fit(self, X, y):
        X_incorrect_type = isinstance(X, (NeighborsBase, BallTree, KDTree))

        if not X_incorrect_type:
            X, y = self._validate_data(X, y, accept_sparse="csr", multi_output=True)
            numeric_type = True if np.issubdtype(X.dtype, np.number) and np.issubdtype(y.dtype, np.number) else False
            single_output = False if y.ndim > 1 and y.shape[1] > 1 else True

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        if daal_check_version(((2020,'P', 3),(2021,'B', 110))) and not X_incorrect_type \
        and self.weights in ['uniform', 'distance'] and self.algorithm in ['brute', 'kd_tree', 'auto'] \
        and (self.metric == 'minkowski' and self.p == 2 or self.metric == 'euclidean') \
        and single_output and fptype is not None and not sp.issparse(X) and numeric_type:
            logging.info("sklearn.neighbors.KNeighborsClassifier.fit: " + method_uses_daal)

            self.outputs_2d_ = False
            check_classification_targets(y)

            # Encode labels
            le = LabelEncoder()
            le.fit(y)
            self.classes_ = le.classes_
            self._y = le.transform(y)

            n_classes = len(self.classes_)
            if n_classes < 2:
                raise ValueError("Training data only contain information about one class.")

            self._y = self._y.ravel()
            self.n_samples_fit_ = X.shape[0]
            self.n_features_in_ = X.shape[1]
            self.effective_metric_ = 'euclidean'
            self._fit_method = self.algorithm
            self._fit_X = X

            params = {
                'method': 'defaultDense',
                'k': self.n_neighbors,
                'nClasses': n_classes,
                'voteWeights': 'voteUniform' if self.weights == 'uniform' else 'voteDistance',
                'resultsToEvaluate': 'computeClassLabels',
                'resultsToCompute': ''
            }

            method = parse_auto_method(self, self.algorithm, self.n_samples_fit_, self.n_features_in_)
            train_alg = training_algorithm(method, fptype, params)
            self.daal_model_ = train_alg.compute(X, self._y.reshape(y.shape[0], 1)).model
            return self
        logging.info("sklearn.neighbors.KNeighborsClassifier.fit: " + method_uses_sklearn)
        return super(KNeighborsClassifier, self).fit(X, y)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        n_features = getattr(self, 'n_features_in_', None)
        shape = getattr(X, 'shape', None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError('Input data shape {} is inconsistent with the trained model'.format(X.shape))

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        if daal_check_version(((2020,'P', 3),(2021,'B', 110))) and hasattr(self, 'daal_model_') \
        and self.weights in ['uniform', 'distance'] and self.algorithm in ['brute', 'kd_tree', 'auto'] \
        and (self.metric == 'minkowski' and self.p == 2 or self.metric == 'euclidean') \
        and self._y.ndim == 1 and fptype is not None and not sp.issparse(X):
            logging.info("sklearn.neighbors.KNeighborsClassifier.predict: " + method_uses_daal)

            params = {
                'method': 'defaultDense',
                'k': self.n_neighbors,
                'nClasses': len(self.classes_),
                'voteWeights': 'voteUniform' if self.weights == 'uniform' else 'voteDistance',
                'resultsToEvaluate': 'computeClassLabels',
                'resultsToCompute': ''
            }

            method = parse_auto_method(self, self.algorithm, self.n_samples_fit_, n_features)
            predict_alg = prediction_algorithm(method, fptype, params)
            prediction_result = predict_alg.compute(X, self.daal_model_)

            # Decode labels
            le = LabelEncoder()
            le.classes_ = self.classes_
            return le.inverse_transform(prediction_result.prediction.ravel().astype(self._y.dtype)).astype(self.classes_[0].dtype)
        logging.info("sklearn.neighbors.KNeighborsClassifier.predict: " + method_uses_sklearn)
        return super(KNeighborsClassifier, self).predict(X)
