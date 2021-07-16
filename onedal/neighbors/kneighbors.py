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

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import is_classifier, is_regressor
from abc import ABCMeta, abstractmethod
from enum import Enum
import sys
from numbers import Number
import warnings

import numpy as np
from scipy import sparse as sp
from ..datatypes import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _check_is_fitted,
    _column_or_1d,
    _check_n_features
)

try:
    import onedal._onedal_py_dpc as backend
except ImportError:
    import onedal._onedal_py_host as backend

def training_algorithm(method, fptype, params):
    if method == 'brute':
        train_alg = d4p.bf_knn_classification_training

    else:
        train_alg = d4p.kdtree_knn_classification_training

    params['fptype'] = fptype
    return train_alg(**params)

def daal4py_fit(estimator, X, fptype):
    estimator._fit_X = X
    estimator._fit_method = estimator.algorithm
    estimator.effective_metric_ = 'euclidean'
    estimator._tree = None
    weights = getattr(estimator, 'weights', 'uniform')

    params = {
        'method': 'defaultDense',
        'k': estimator.n_neighbors,
        'voteWeights': 'voteUniform' if weights == 'uniform' else 'voteDistance',
        'resultsToCompute': 'computeIndicesOfNeighbors|computeDistances',
        'resultsToEvaluate': 'none' if getattr(estimator, '_y', None) is None
        else 'computeClassLabels'
    }
    if hasattr(estimator, 'classes_'):
        params['nClasses'] = len(estimator.classes_)

    if getattr(estimator, '_y', None) is None:
        labels = None
    else:
        labels = estimator._y.reshape(-1, 1)

    method = parse_auto_method(
        estimator, estimator.algorithm,
        estimator.n_samples_fit_, estimator.n_features_in_)
    estimator._fit_method = method
    train_alg = training_algorithm(method, fptype, params)
    estimator._daal_model = train_alg.compute(X, labels).model

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

    if sklearn_check_version("0.23") and check_params.get('ensure_2d', True):
        estimator._check_n_features(X, reset=reset)

    return out

class NeighborsBase(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
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

        X_incorrect_type = isinstance(
            X, (KDTree, BallTree, NeighborsBase))

        single_output = True
        self._daal_model = None
        shape = None
        correct_n_classes = True

        try:
            requires_y = self._get_tags()["requires_y"]
        except KeyError:
            requires_y = False

        if y is not None or requires_y:
            if not X_incorrect_type or y is None:
                X, y = validate_data(
                    self, X, y, accept_sparse="csr", multi_output=True,
                    dtype=[np.float64, np.float32])
                single_output = False if y.ndim > 1 and y.shape[1] > 1 else True

            shape = y.shape

            if is_classifier(self) or is_regressor(self):
                if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
                    self.outputs_2d_ = False
                    y = y.reshape((-1, 1))
                else:
                    self.outputs_2d_ = True

                if is_classifier(self):
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
                    correct_n_classes = False
            else:
                self._y = y
        else:
            if not X_incorrect_type:
                X, _ = validate_data(
                    self, X, accept_sparse='csr', dtype=[np.float64, np.float32])

        if not X_incorrect_type:
            self.n_samples_fit_ = X.shape[0]
            self.n_features_in_ = X.shape[1]

        try:
            fptype = getFPType(X)
        except ValueError:
            fptype = None

        weights = getattr(self, 'weights', 'uniform')

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    "Expected n_neighbors > 0. Got %d" %
                    self.n_neighbors
                )
            if not isinstance(self.n_neighbors, numbers.Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(self.n_neighbors))

        condition = (self.metric == 'minkowski' and self.p == 2) or \
            self.metric == 'euclidean'

        daal4py_fit(self, X, fptype)
        result = self

        if y is not None and is_regressor(self):
            self._y = y if shape is None else y.reshape(shape)

        return result


# class KNeighborsMixin(BaseKNeighborsMixin):
#     def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
#         daal_model = getattr(self, '_daal_model', None)
#         if X is not None:
#             X = _check_array(
#                 X, accept_sparse='csr', dtype=[
#                     np.float64, np.float32])
#         x = self._fit_X if X is None else X
#         try:
#             fptype = getFPType(x)
#         except ValueError:
#             fptype = None

#         if daal_model is not None and fptype is not None and not sp.issparse(
#                 X):
#             logging.info(
#                 "sklearn.neighbors.KNeighborsMixin."
#                 "kneighbors: " + get_patch_message("daal"))
#             result = daal4py_kneighbors(self, X, n_neighbors, return_distance)
#         else:
#             logging.info(
#                 "sklearn.neighbors.KNeighborsMixin."
#                 "kneighbors:" + get_patch_message("sklearn"))
#             if daal_model is not None or getattr(self, '_tree', 0) is None and \
#                     self._fit_method == 'kd_tree':
#                 if sklearn_check_version("0.24"):
#                     BaseNeighborsBase._fit(self, self._fit_X, getattr(self, '_y', None))
#                 else:
#                     BaseNeighborsBase._fit(self, self._fit_X)
#             result = super(KNeighborsMixin, self).kneighbors(
#                 X, n_neighbors, return_distance)

#         return result


# class RadiusNeighborsMixin(BaseRadiusNeighborsMixin):
#     def radius_neighbors(self, X=None, radius=None, return_distance=True,
#                          sort_results=False):
#         daal_model = getattr(self, '_daal_model', None)

#         if daal_model is not None or getattr(self, '_tree', 0) is None and \
#                 self._fit_method == 'kd_tree':
#             if sklearn_check_version("0.24"):
#                 BaseNeighborsBase._fit(self, self._fit_X, getattr(self, '_y', None))
#             else:
#                 BaseNeighborsBase._fit(self, self._fit_X)
#         if sklearn_check_version("0.22"):
#             result = BaseRadiusNeighborsMixin.radius_neighbors(
#                 self, X, radius, return_distance, sort_results)
#         else:
#             result = BaseRadiusNeighborsMixin.radius_neighbors(
#                 self, X, radius, return_distance)

#         return result
