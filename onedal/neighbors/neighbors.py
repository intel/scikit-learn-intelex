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
    _convert_to_supported,
    _num_samples
)

from daal4py import (
    bf_knn_classification_training,
    bf_knn_classification_prediction,
    kdtree_knn_classification_training,
    kdtree_knn_classification_prediction
)
from onedal import _backend

from ..common._mixin import ClassifierMixin, RegressorMixin
from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted, _is_classifier, _is_regressor
from ..datatypes._data_conversion import from_table, to_table


class NeighborsCommonBase(metaclass=ABCMeta):
    def _get_policy(self, queue, *data):
        return _get_policy(queue, *data)

    def _parse_auto_method(self, method, n_samples, n_features):
        result_method = method

        if (method in ['auto', 'ball_tree']):
            condition = self.n_neighbors is not None and \
                self.n_neighbors >= n_samples // 2
            if self.metric == 'precomputed' or n_features > 15 or condition:
                result_method = 'brute'
            else:
                if self.metric == 'euclidean':
                    result_method = 'kd_tree'
                else:
                    result_method = 'brute'

        return result_method

    def _validate_data(self, X, y=None, reset=True,
                       validate_separately=False, **check_params):
        if y is None:
            if self.requires_y:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
            X = _check_array(X, **check_params)
            out = X, y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling _check_array()
                # on X and y isn't equivalent to just calling _check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = _check_array(X, **check_X_params)
                y = _check_array(y, **check_y_params)
            else:
                X, y = _check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            _check_n_features(self, X, reset=reset)

        return out

    def _get_weights(self, dist, weights):
        if weights in (None, "uniform"):
            return None
        if weights == "distance":
            # if user attempts to classify a point that was zero distance from one
            # or more training points, those training points are weighted as 1.0
            # and the other points as 0.0
            if dist.dtype is np.dtype(object):
                for point_dist_i, point_dist in enumerate(dist):
                    # check if point_dist is iterable
                    # (ex: RadiusNeighborClassifier.predict may set an element of
                    # dist to 1e-6 to represent an 'outlier')
                    if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                        dist[point_dist_i] = point_dist == 0.0
                    else:
                        dist[point_dist_i] = 1.0 / point_dist
            else:
                with np.errstate(divide="ignore"):
                    dist = 1.0 / dist
                inf_mask = np.isinf(dist)
                inf_row = np.any(inf_mask, axis=1)
                dist[inf_row] = inf_mask[inf_row]
            return dist
        elif callable(weights):
            return weights(dist)
        else:
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )

    def _get_onedal_params(self, X, y=None):
        class_count = 0 if self.classes_ is None else len(self.classes_)
        weights = getattr(self, 'weights', 'uniform')
        return {
            'fptype': 'float' if X.dtype == np.float32 else 'double',
            'vote_weights': 'uniform' if weights == 'uniform' else 'distance',
            'method': self._fit_method,
            'radius': self.radius,
            'class_count': class_count,
            'neighbor_count': self.n_neighbors,
            'metric': self.effective_metric_,
            'p': self.p,
            'metric_params': self.effective_metric_params_,
            'result_option': 'indices|distances' if y is None else 'responses',
        }

    def _get_daal_params(self, data):
        class_count = 0 if self.classes_ is None else len(self.classes_)
        weights = getattr(self, 'weights', 'uniform')
        params = {
            'fptype': 'float' if data.dtype == np.float32 else 'double',
            'method': 'defaultDense',
            'k': self.n_neighbors,
            'voteWeights': 'voteUniform' if weights == 'uniform' else 'voteDistance',
            'resultsToCompute': 'computeIndicesOfNeighbors|computeDistances',
            'resultsToEvaluate': 'none'
            if getattr(self, '_y', None) is None or _is_regressor(self)
            else 'computeClassLabels'
        }
        if class_count != 0:
            params['nClasses'] = class_count
        return params


class NeighborsBase(NeighborsCommonBase, metaclass=ABCMeta):
    def __init__(self, n_neighbors=None, radius=None,
                 algorithm='auto', metric='minkowski',
                 p=2, metric_params=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.metric_params = metric_params

    def _validate_targets(self, y, dtype):
        arr = _column_or_1d(y, warn=True)

        try:
            return arr.astype(dtype, copy=False)
        except ValueError:
            return arr

    def _validate_n_classes(self):
        if len(self.classes_) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                " class" % len(self.classes_))

    def _fit(self, X, y, queue):
        self._onedal_model = None
        self._tree = None
        self._shape = None
        self.classes_ = None
        self.effective_metric_ = getattr(self, 'effective_metric_', self.metric)
        self.effective_metric_params_ = getattr(
            self, 'effective_metric_params_', self.metric_params)

        if y is not None or self.requires_y:
            shape = getattr(y, 'shape', None)
            X, y = super()._validate_data(X, y, dtype=[np.float64, np.float32])
            self._shape = shape if shape is not None else y.shape

            if _is_classifier(self):
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                _check_classification_targets(y)
                self.classes_ = []
                self._y = np.empty(y.shape, dtype=int)
                for k in range(self._y.shape[1]):
                    classes, self._y[:, k] = np.unique(
                        y[:, k], return_inverse=True)
                    self.classes_.append(classes)

                if not self.outputs_2d_:
                    self.classes_ = self.classes_[0]
                    self._y = self._y.ravel()

                self._validate_n_classes()
            else:
                self._y = y
        else:
            X, _ = super()._validate_data(X, dtype=[np.float64, np.float32])

        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self._fit_X = X

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    "Expected n_neighbors > 0. Got %d" %
                    self.n_neighbors
                )
            if not isinstance(self.n_neighbors, Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(self.n_neighbors))

        self._fit_method = super()._parse_auto_method(
            self.algorithm,
            self.n_samples_fit_, self.n_features_in_)

        _fit_y = None
        gpu_device = queue is not None and queue.sycl_device.is_gpu

        if _is_classifier(self) or (_is_regressor(self) and gpu_device):
            _fit_y = self._validate_targets(self._y, X.dtype).reshape((-1, 1))
        result = self._onedal_fit(X, _fit_y, queue)

        if y is not None and _is_regressor(self):
            self._y = y if self._shape is None else y.reshape(self._shape)

        self._onedal_model = result
        result = self

        return result

    def _kneighbors(self, X=None, n_neighbors=None,
                    return_distance=True, queue=None):
        n_features = getattr(self, 'n_features_in_', None)
        shape = getattr(X, 'shape', None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError((f'X has {X.shape[1]} features, '
                              f'but kneighbors is expecting '
                              f'{n_features} features as input'))

        _check_is_fitted(self)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(
                "Expected n_neighbors > 0. Got %d" %
                n_neighbors
            )
        else:
            if not isinstance(n_neighbors, Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(n_neighbors))

        if X is not None:
            query_is_train = False
            X = _check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1
        self.n_neighbors = n_neighbors

        n_samples_fit = self.n_samples_fit_
        if n_neighbors > n_samples_fit:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (n_samples_fit, n_neighbors)
            )

        chunked_results = None
        method = super()._parse_auto_method(
            self._fit_method, self.n_samples_fit_, n_features)

        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            params = super()._get_daal_params(X)
        else:
            params = super()._get_onedal_params(X)

        prediction_results = self._onedal_predict(
            self._onedal_model, X, params, queue=queue)

        if self.effective_metric_ == 'euclidean' and not gpu_device:
            distances = prediction_results.distances
            indices = prediction_results.indices
        else:
            distances = from_table(prediction_results.distances)
            indices = from_table(prediction_results.indices)

        if method == 'kd_tree':
            for i in range(distances.shape[0]):
                seq = distances[i].argsort()
                indices[i] = indices[i][seq]
                distances[i] = distances[i][seq]

        if return_distance:
            results = distances, indices
        else:
            results = indices

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
        distances = distances[:, 1:]
        indices = indices[:, 1:]

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


class KNeighborsClassifier(NeighborsBase, ClassifierMixin):
    def __init__(self, n_neighbors=5, *,
                 weights='uniform', algorithm='auto',
                 p=2, metric='minkowski', metric_params=None, **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric, p=p,
            metric_params=metric_params,
            **kwargs)
        self.weights = weights

    def _get_onedal_params(self, X, y=None):
        params = super()._get_onedal_params(X, y)
        return params

    def _get_daal_params(self, data):
        params = super()._get_daal_params(data)
        params['resultsToEvaluate'] = 'computeClassLabels'
        params['resultsToCompute'] = ''
        return params

    def _onedal_fit(self, X, y, queue):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            params = self._get_daal_params(X)
            if self._fit_method == 'brute':
                train_alg = bf_knn_classification_training

            else:
                train_alg = kdtree_knn_classification_training

            return train_alg(**params).compute(X, y).model

        policy = self._get_policy(queue, X, y)
        X, y = _convert_to_supported(policy, X, y)
        params = self._get_onedal_params(X, y)
        train_alg = _backend.neighbors.classification.train(policy, params,
                                                            *to_table(X, y))

        return train_alg.model

    def _onedal_predict(self, model, X, params, queue):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            if self._fit_method == 'brute':
                predict_alg = bf_knn_classification_prediction

            else:
                predict_alg = kdtree_knn_classification_prediction

            return predict_alg(**params).compute(X, model)

        policy = self._get_policy(queue, X)
        X = _convert_to_supported(policy, X)
        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(_backend.neighbors.classification)
        if 'responses' not in params['result_option']:
            params['result_option'] += '|responses'
        params['fptype'] = 'float' if X.dtype == np.float32 else 'double'
        result = _backend.neighbors.classification.infer(
            policy, params, model, to_table(X))

        return result

    def fit(self, X, y, queue=None):
        return super()._fit(X, y, queue=queue)

    def predict(self, X, queue=None):
        X = _check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        onedal_model = getattr(self, '_onedal_model', None)
        n_features = getattr(self, 'n_features_in_', None)
        n_samples_fit_ = getattr(self, 'n_samples_fit_', None)
        shape = getattr(X, 'shape', None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError((f'X has {X.shape[1]} features, '
                              f'but KNNClassifier is expecting '
                              f'{n_features} features as input'))

        _check_is_fitted(self)

        self._fit_method = super()._parse_auto_method(
            self.algorithm,
            n_samples_fit_, n_features)

        self._validate_n_classes()

        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            params = self._get_daal_params(X)
        else:
            params = self._get_onedal_params(X)

        prediction_result = self._onedal_predict(onedal_model, X, params, queue=queue)
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            responses = prediction_result.prediction
        else:
            responses = from_table(prediction_result.responses)
        result = self.classes_.take(
            np.asarray(responses.ravel(), dtype=np.intp))

        return result

    def predict_proba(self, X, queue=None):
        neigh_dist, neigh_ind = self.kneighbors(X, queue=queue)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X)

        weights = self._get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)

        all_rows = np.arange(n_queries)
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

    def kneighbors(self, X=None, n_neighbors=None,
                   return_distance=True, queue=None):
        return super()._kneighbors(X, n_neighbors, return_distance, queue=queue)


class KNeighborsRegressor(NeighborsBase, RegressorMixin):
    def __init__(self, n_neighbors=5, *,
                 weights='uniform', algorithm='auto',
                 p=2, metric='minkowski', metric_params=None, **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric, p=p,
            metric_params=metric_params,
            **kwargs)
        self.weights = weights

    def _get_onedal_params(self, X, y=None):
        params = super()._get_onedal_params(X, y)
        return params

    def _get_daal_params(self, data):
        params = super()._get_daal_params(data)
        params['resultsToCompute'] = 'computeIndicesOfNeighbors|computeDistances'
        params['resultsToEvaluate'] = 'none'
        return params

    def _onedal_fit(self, X, y, queue):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            params = self._get_daal_params(X)
            if self._fit_method == 'brute':
                train_alg = bf_knn_classification_training

            else:
                train_alg = kdtree_knn_classification_training

            return train_alg(**params).compute(X, y).model

        policy = self._get_policy(queue, X, y)
        X, y = _convert_to_supported(policy, X, y)
        params = self._get_onedal_params(X, y)
        train_alg_regr = _backend.neighbors.regression.train
        train_alg_srch = _backend.neighbors.search.train

        if gpu_device:
            return train_alg_regr(policy, params, *to_table(X, y)).model
        return train_alg_srch(policy, params, to_table(X)).model

    def _onedal_predict(self, model, X, params, queue):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            if self._fit_method == 'brute':
                predict_alg = bf_knn_classification_prediction

            else:
                predict_alg = kdtree_knn_classification_prediction

            return predict_alg(**params).compute(X, model)

        policy = self._get_policy(queue, X)
        X = _convert_to_supported(policy, X)
        backend = _backend.neighbors.regression if gpu_device \
            else _backend.neighbors.search

        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(backend)
        params['fptype'] = 'float' if X.dtype == np.float32 else 'double'
        result = backend.infer(policy, params, model, to_table(X))

        return result

    def fit(self, X, y, queue=None):
        return super()._fit(X, y, queue=queue)

    def kneighbors(self, X=None, n_neighbors=None,
                   return_distance=True, queue=None):
        return super()._kneighbors(X, n_neighbors, return_distance, queue=queue)

    def _predict_gpu(self, X, queue=None):
        X = _check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
        onedal_model = getattr(self, '_onedal_model', None)
        n_features = getattr(self, 'n_features_in_', None)
        n_samples_fit_ = getattr(self, 'n_samples_fit_', None)
        shape = getattr(X, 'shape', None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError((f'X has {X.shape[1]} features, '
                              f'but KNNClassifier is expecting '
                              f'{n_features} features as input'))

        _check_is_fitted(self)

        self._fit_method = super()._parse_auto_method(
            self.algorithm,
            n_samples_fit_, n_features)

        params = self._get_onedal_params(X)

        prediction_result = self._onedal_predict(onedal_model, X, params, queue=queue)
        responses = from_table(prediction_result.responses)
        result = responses.ravel()

        return result

    def _predict_skl(self, X, queue=None):
        neigh_dist, neigh_ind = self.kneighbors(X, queue=queue)

        weights = self._get_weights(neigh_dist, self.weights)

        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom

        if self._y.ndim == 1:
            y_pred = y_pred.ravel()

        return y_pred

    def predict(self, X, queue=None):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        is_uniform_weights = getattr(self, 'weights', 'uniform') == 'uniform'
        return self._predict_gpu(X, queue=queue) \
            if gpu_device and is_uniform_weights else self._predict_skl(X, queue=queue)


class NearestNeighbors(NeighborsBase):
    def __init__(self, n_neighbors=5, *,
                 weights='uniform', algorithm='auto',
                 p=2, metric='minkowski', metric_params=None, **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric, p=p,
            metric_params=metric_params,
            **kwargs)
        self.weights = weights

    def _get_onedal_params(self, X, y=None):
        params = super()._get_onedal_params(X, y)
        return params

    def _get_daal_params(self, data):
        params = super()._get_daal_params(data)
        params['resultsToCompute'] = 'computeIndicesOfNeighbors|computeDistances'
        params['resultsToEvaluate'] = 'none' if getattr(self, '_y', None) is None \
            else 'computeClassLabels'
        return params

    def _onedal_fit(self, X, y, queue):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            params = self._get_daal_params(X)
            if self._fit_method == 'brute':
                train_alg = bf_knn_classification_training

            else:
                train_alg = kdtree_knn_classification_training

            return train_alg(**params).compute(X, y).model

        policy = self._get_policy(queue, X, y)
        X, y = _convert_to_supported(policy, X, y)
        params = self._get_onedal_params(X, y)
        train_alg = _backend.neighbors.search.train(policy, params,
                                                    to_table(X))

        return train_alg.model

    def _onedal_predict(self, model, X, params, queue):
        gpu_device = queue is not None and queue.sycl_device.is_gpu
        if self.effective_metric_ == 'euclidean' and not gpu_device:
            if self._fit_method == 'brute':
                predict_alg = bf_knn_classification_prediction

            else:
                predict_alg = kdtree_knn_classification_prediction

            return predict_alg(**params).compute(X, model)

        policy = self._get_policy(queue, X)
        X = _convert_to_supported(policy, X)
        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(_backend.neighbors.search)

        params['fptype'] = 'float' if X.dtype == np.float32 else 'double'
        result = _backend.neighbors.search.infer(policy, params, model, to_table(X))

        return result

    def fit(self, X, y, queue=None):
        return super()._fit(X, y, queue=queue)

    def kneighbors(self, X=None, n_neighbors=None,
                   return_distance=True, queue=None):
        return super()._kneighbors(X, n_neighbors, return_distance, queue=queue)
