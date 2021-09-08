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

# from sklearn.utils.multiclass import check_classification_targets
from abc import ABCMeta, abstractmethod
from enum import Enum
import sys
from numbers import Number, Integral
import warnings

from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version

from sklearn.base import is_classifier, is_regressor
# from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import ClassifierMixin as BaseClassifierMixin
from sklearn.neighbors._base import KNeighborsMixin as BaseKNeighborsMixin
from sklearn.neighbors._base import RadiusNeighborsMixin as BaseRadiusNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as BaseNeighborsBase
from sklearn.neighbors._kd_tree import KDTree

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _check_is_fitted,
    _column_or_1d,
    _check_n_features,
    _check_classification_targets
)

try:
    import onedal._onedal_py_dpc as backend
except ImportError:
    import onedal._onedal_py_host as backend

from ..common._policy import _HostPolicy

def parse_auto_method(estimator, method, n_samples, n_features):
    result_method = method

    if (method in ['auto', 'ball_tree']):
        condition = estimator.n_neighbors is not None and \
            estimator.n_neighbors >= estimator.n_samples_fit_ // 2
        if estimator.metric == 'precomputed' or n_features > 11 or condition:
            result_method = 'brute'
        else:
            # if estimator.metric in KDTree.valid_metrics:
            if estimator.metric in 'euclidean':
                result_method = 'kd_tree'
            else:
                result_method = 'brute'

    return result_method

def validate_data(estimator, X, y=None, reset=True,
                  validate_separately=False, **check_params):
    if y is None:
        try:
            requires_y = estimator._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        if requires_y:
            raise ValueError(
                f"This {estimator.__class__.__name__} estimator "
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

    if LooseVersion(sklearn_version) >= LooseVersion("0.23") and check_params.get('ensure_2d', True):
        estimator._check_n_features(X, reset=reset)

    return out

def _onedal_fit(estimator, X, y):
    policy = _HostPolicy()
    params = estimator._get_onedal_params(X)
    train_alg = backend.neighbors.classification.train(policy, params,
                            backend.from_numpy(X),
                            backend.from_numpy(y))

    return train_alg

def _onedal_predict(estimator, model, X):
    policy = _HostPolicy()
    params = estimator._get_onedal_params(X)

    if hasattr(estimator, '_onedal_model'):
        model = estimator._onedal_model
    else:
        model = estimator._create_model(backend.neighbors.classification)
    result = backend.neighbors.classification.infer(policy, params, model, backend.from_numpy(X))

    return result

def _kneighbors(estimator, X=None, n_neighbors=None,
                    return_distance=True):
    n_features = getattr(estimator, 'n_features_in_', None)
    shape = getattr(X, 'shape', None)
    if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        raise ValueError((f'X has {X.shape[1]} features, '
                        f'but kneighbors is expecting {n_features} features as input'))

    _check_is_fitted(estimator)

    if n_neighbors is None:
        n_neighbors = estimator.n_neighbors
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

    method = parse_auto_method(
        estimator, estimator._fit_method, estimator.n_samples_fit_, n_features)

    prediction_results = _onedal_predict(estimator, estimator._onedal_model, X)

    distances = backend.to_numpy(prediction_results.distances)
    indices = backend.to_numpy(prediction_results.indices)

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

class NeighborsBase(BaseNeighborsBase, metaclass=ABCMeta):
    def __init__(self, n_neighbors=None, radius=None,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.radius=radius
        self.algorithm=algorithm
        self.leaf_size=leaf_size
        self.metric=metric
        self.p=p
        self.metric_params=metric_params
        self.n_jobs=n_jobs

    def _validate_targets(self, y, dtype):
        self.classes_ = None
        return _column_or_1d(y, warn=True).astype(dtype, copy=False)

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
            'leaf_size': self.leaf_size,
            'metric': self.metric,
            'p': self.p,
            'metric_params': self.metric_params,
            'n_jobs': self.n_jobs,
            #TODO create autodispatching for this
            'result_option': 'all',
        }

    def _validate_n_classes(self, stage='train'):
        if len(self.classes_) < 2:
            raise ValueError((f"Classifier can't {stage} when only one class is present."))

    def _fit(self, X, y=None):
        if self.metric_params is not None and 'p' in self.metric_params:
            if self.p is not None:
                warnings.warn("Parameter p is found in metric_params. "
                              "The corresponding parameter from __init__ "
                              "is ignored.", SyntaxWarning, stacklevel=2)
            effective_p = self.metric_params["p"]
        else:
            effective_p = self.p

        if self.metric in ["minkowski"] and effective_p < 1:
            raise ValueError("p must be greater or equal to one for minkowski metric")

        self._onedal_model = None
        shape = None

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        if y is not None or requires_y:
            X, y = validate_data(
                self, X, y, accept_sparse="csr",# multi_output=True,
                dtype=[np.float64, np.float32])
            y = self._validate_targets(y, X.dtype)
            shape = y.shape

            if is_classifier(self) or is_regressor(self):
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                if is_classifier(self):
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
            else:
                self._y = y
        else:
            X, _ = validate_data(
                self, X, accept_sparse='csr', dtype=[np.float64, np.float32])
            y = self._validate_targets(y, X.dtype)

        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]

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

        self._validate_n_classes()

        method = parse_auto_method(
            self, self.algorithm,
            self.n_samples_fit_, self.n_features_in_)
        self._fit_method = method

        result = _onedal_fit(self, X, y)

        if y is not None and is_regressor(self):
            self._y = y if shape is None else y.reshape(shape)
        
        self._onedal_model = result.model
        result = self

        return result

class KNeighborsClassifier(NeighborsBase, BaseClassifierMixin):
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
        self.weights = weights
        # self._estimator_type = getattr(self, "_estimator_type", "classifier")

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
            'leaf_size': self.leaf_size,
            'metric': self.metric,
            'p': self.p,
            'metric_params': self.metric_params,
            'n_jobs': self.n_jobs,
            'result_option': 'responses',
        }

    def fit(self, X, y):
        return super()._fit(X, y)

    def predict(self, X):
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

        method = parse_auto_method(
            self, self.algorithm,
            n_samples_fit_, n_features)
        self._fit_method = method

        self._validate_n_classes('predict')

        prediction_result = _onedal_predict(self, onedal_model, X)
        responses = backend.to_numpy(prediction_result.responses)
        result = self.classes_.take(
            np.asarray(responses.ravel(), dtype=np.intp))

        return result

class KNeighborsMixin(BaseKNeighborsMixin):
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        onedal_model = getattr(self, '_onedal_model', None)
        if X is not None:
            X = _check_array(
                X, accept_sparse='csr', dtype=[
                    np.float64, np.float32])

        if onedal_model is not None and not sp.issparse(X):
            result = _kneighbors(self, X, n_neighbors, return_distance)

        return result
