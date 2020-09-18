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
from .._utils import getFPType, daal_check_version, method_uses_sklearn, method_uses_daal
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.neighbors._base import KNeighborsMixin as BaseKNeighborsMixin
from sklearn.neighbors._classification import KNeighborsClassifier as BaseKNeighborsClassifier
from joblib import effective_n_jobs
from sklearn.neighbors._base import _check_precomputed
import logging


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

        if daal_check_version((2020, 3)) and self._fit_method in ['brute', 'kd_tree', 'auto'] \
        and (self.effective_metric_ == 'minkowski' and self.p == 2 or self.effective_metric_ == 'euclidean') \
        and fptype is not None and not sp.issparse(X):
            logging.info("sklearn.neighbors.KNeighborsMixin.kneighbors: " + method_uses_daal)

            method = self._fit_method

            if (method == 'auto'):
                if    (n_features >= 13)
                   or (n_features == 8  and n_samples_fit <= 10000)
                   or (n_features == 9  and n_samples_fit <= 20000)
                   or (n_features == 10 and n_samples_fit <= 50000)
                   or (n_features == 11 and n_samples_fit <= 1000000)
                   or (n_features == 12 and n_samples_fit <= 2000000):
                    method = 'brute'
                else:
                    method = 'kd_tree'

            if method == 'brute':
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
                'resultsToCompute': 'computeIndicesOfNeightbors',
                'resultsToEvaluate': 'none'
            }
            if return_distance:
                alg_params['resultsToCompute'] += '|computeDistances'

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


class KNeighborsClassifier(BaseKNeighborsClassifier, KNeighborsMixin):
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

        if daal_check_version((2020, 3)) \
        and self.weights in ['uniform', 'distance'] and self.algorithm in ['brute', 'kd_tree', 'auto'] \
        and (self.metric == 'minkowski' and self.p == 2 or self.metric == 'euclidean') \
        and self._y.ndim == 1 and fptype is not None and not sp.issparse(X):
            logging.info("sklearn.neighbors.KNeighborsClassifier.predict: " + method_uses_daal)

            n_classes = len(self.classes_)

            method = self.algorithm

            if (method == 'auto'):
                if    (n_features > 13)
                   or (n_features == 8  and self.n_samples_fit_ <= 10000)
                   or (n_features == 9  and self.n_samples_fit_ <= 20000)
                   or (n_features == 10 and self.n_samples_fit_ <= 50000)
                   or (n_features == 11 and self.n_samples_fit_ <= 1000000)
                   or (n_features == 12 and self.n_samples_fit_ <= 2000000)
                   or (n_features == 13 and self.n_samples_fit_ <= 5000000):
                    method = 'brute'
                else:
                    method = 'kd_tree'

            if method == 'brute':
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
                'resultsToEvaluate': 'computeClassLabels',
                'resultsToCompute': ''
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
            logging.info("sklearn.neighbors.KNeighborsClassifier.predict: " + method_uses_sklearn)
            return super(KNeighborsClassifier, self).predict(X)
