#===============================================================================
# Copyright 2021 Intel Corporation
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

from abc import ABCMeta, abstractmethod
from enum import Enum
import sys
from numbers import Number, Integral
import warnings

from distutils.version import LooseVersion

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _check_X_y,
    _check_array,
    _column_or_1d,
    _check_n_features,
    _check_classification_targets,
    _num_samples
)

from onedal import _backend

from ..common._mixin import ClassifierMixin, RegressorMixin
from ..common._policy import _get_policy
from ..common._estimator_checks import _check_is_fitted, _is_classifier, _is_regressor
from ..datatypes._data_conversion import from_table, to_table


class NeighborsCommonBase(metaclass=ABCMeta):
    def _parse_auto_method(self, method, n_samples, n_features):
        result_method = method

        if (method in ['auto', 'ball_tree']):
            condition = self.n_neighbors is not None and \
                self.n_neighbors >= self.n_samples_fit_ // 2
            if self.metric == 'precomputed' or n_features > 11 or condition:
                result_method = 'brute'
            else:
                if self.metric in 'euclidean':
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
        elif weights == "distance":
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

    def _get_onedal_params(self, data):
        class_count = 0 if self.classes_ is None else len(self.classes_)
        weights = getattr(self, 'weights', 'uniform')
        return {
            'fptype': 'float' if data.dtype is np.dtype('float32') else 'double',
            'vote_weights': 'uniform' if weights == 'uniform' else 'distance',
            'method': self._fit_method,
            'radius': self.radius,
            'class_count': class_count,
            'neighbor_count': self.n_neighbors,
            'metric': self.metric,
            'p': self.p,
            'metric_params': self.metric_params,
            'result_option': 'indices|distances',
        }

class NeighborsBase(NeighborsCommonBase, metaclass=ABCMeta):
    def __init__(self, n_neighbors=None, radius=None,
                 algorithm='auto', metric='minkowski',
                 p=2, metric_params=None):
        self.n_neighbors = n_neighbors
        self.radius=radius
        self.algorithm=algorithm
        self.metric=metric
        self.p=p
        self.metric_params=metric_params

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
        if self.metric_params is not None and 'p' in self.metric_params:
            if self.p is not None:
                warnings.warn("Parameter p is found in metric_params. "
                              "The corresponding parameter from __init__ "
                              "is ignored.", SyntaxWarning, stacklevel=2)
            self.effective_metric_params_ = self.metric_params.copy()
            effective_p = self.metric_params["p"]
        else:
            self.effective_metric_params_ = {}
            effective_p = self.p

        if self.metric in ["minkowski"]:
            if effective_p < 1:
                raise ValueError("p must be greater or equal to one for minkowski metric")
            self.effective_metric_params_["p"] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == "minkowski":
            p = self.effective_metric_params_.pop("p", 2)
            if p < 1:
                raise ValueError(
                    "p must be greater or equal to one for minkowski metric"
                )
            elif p == 1:
                self.effective_metric_ = "manhattan"
            elif p == 2:
                self.effective_metric_ = "euclidean"
            elif p == np.inf:
                self.effective_metric_ = "chebyshev"
            else:
                self.effective_metric_params_["p"] = p

        self._onedal_model = None
        self._tree = None
        self.shape = None
        self.classes_ = None

        if y is not None or self.requires_y:
            X, y = super()._validate_data(X, y, dtype=[np.float64, np.float32])
            self.shape = y.shape

            if _is_classifier(self) or _is_regressor(self):
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                if _is_classifier(self):
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

        if _is_classifier(self) and y.dtype not in [np.float64, np.float32]:
            y = self._validate_targets(self._y, X.dtype).reshape((-1, 1))
        result = self._onedal_fit(X, y, queue)

        if y is not None and _is_regressor(self):
            self._y = y if self.shape is None else y.reshape(self.shape)
        
        self._onedal_model = result.model
        result = self

        return result

    def _kneighbors(self, X=None, n_neighbors=None,
                        return_distance=True, queue=None):
        n_features = getattr(self, 'n_features_in_', None)
        shape = getattr(X, 'shape', None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError((f'X has {X.shape[1]} features, '
                            f'but kneighbors is expecting {n_features} features as input'))

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

        params = super()._get_onedal_params(X)

        prediction_results = self._onedal_predict(self._onedal_model, X, params, queue=queue)

        distances = from_table(prediction_results.distances)
        indices = from_table(prediction_results.indices)

        if method == 'kd_tree':
            for i in range(distances.shape[0]):
                seq = distances[i].argsort()
                indices[i] = indices[i][seq]
                distances[i] = distances[i][seq]

        if return_distance:
            results = distances, indices.astype(int)
        else:
            results = indices.astype(int)

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

    def _get_onedal_params(self, data):
        params = super()._get_onedal_params(data)
        params['result_option'] = 'responses'
        return params

    def _onedal_fit(self, X, y, queue):
        policy = _get_policy(queue, X, y)
        params = self._get_onedal_params(X)
        train_alg = _backend.neighbors.classification.train(policy, params,
                                *to_table(X, y))

        return train_alg

    def _onedal_predict(self, model, X, params, queue):
        policy = _get_policy(queue, X)

        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(_backend.neighbors.classification)
        result = _backend.neighbors.classification.infer(policy, params, model, to_table(X))

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

        params = self._get_onedal_params(X)

        prediction_result = self._onedal_predict(onedal_model, X, params, queue=queue)
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

    def _get_onedal_params(self, data):
        params = super()._get_onedal_params(data)
        params['result_option'] = 'indices|distances'
        return params

    def _onedal_fit(self, X, y, queue):
        policy = _get_policy(queue, X, y)
        params = self._get_onedal_params(X)
        train_alg = _backend.neighbors.search.train(policy, params,
                                to_table(X))

        return train_alg

    def _onedal_predict(self, model, X, params, queue):
        policy = _get_policy(queue, X)

        if hasattr(self, '_onedal_model'):
            model = self._onedal_model
        else:
            model = self._create_model(_backend.neighbors.search)
        result = _backend.neighbors.search.infer(policy, params, model, to_table(X))

        return result

    def fit(self, X, y, queue=None):
        return super()._fit(X, y, queue=queue)

    def kneighbors(self, X=None, n_neighbors=None,
                        return_distance=True, queue=None):
        return super()._kneighbors(X, n_neighbors, return_distance, queue=queue)
