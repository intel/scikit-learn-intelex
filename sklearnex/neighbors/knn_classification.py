#!/usr/bin/env python
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

# from daal4py.sklearn.neighbors import KNeighborsClassifier

from scipy import sparse as sp
import logging
from .._utils import get_patch_message

from onedal.datatypes import (
    _validate_targets,
    _check_X_y,
    _check_array,
    _check_is_fitted,
    _column_or_1d,
    _check_n_features
)

from distutils.version import LooseVersion
from sklearn import __version__ as sklearn_version

if LooseVersion(sklearn_version) >= LooseVersion("0.22"):
    from sklearn.neighbors._base import KNeighborsMixin as BaseKNeighborsMixin
    from sklearn.neighbors._base import RadiusNeighborsMixin as BaseRadiusNeighborsMixin
    from sklearn.neighbors._base import NeighborsBase as BaseNeighborsBase
    from sklearn.neighbors._ball_tree import BallTree
    from sklearn.neighbors._kd_tree import KDTree
    from sklearn.neighbors._base import _check_weights
else:
    from sklearn.neighbors.base import KNeighborsMixin as BaseKNeighborsMixin
    from sklearn.neighbors.base import RadiusNeighborsMixin as BaseRadiusNeighborsMixin
    from sklearn.neighbors.base import NeighborsBase as BaseNeighborsBase
    from sklearn.neighbors.ball_tree import BallTree
    from sklearn.neighbors.kd_tree import KDTree
    from sklearn.neighbors.base import _check_weights

from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.multiclass import check_classification_targets

from onedal.neighbors import NeighborsBase as onedal_NeighborsBase
from onedal.neighbors import KNeighborsMixin as onedal_KNeighborsMixin
from onedal.neighbors.neighbors import validate_data, parse_auto_method, _onedal_predict, _get_responses



from sklearn.base import ClassifierMixin as BaseClassifierMixin
from .._utils import get_patch_message
import numpy as np
from scipy import sparse as sp
import logging

if LooseVersion(sklearn_version) >= LooseVersion("0.22"):
    from sklearn.neighbors._classification import KNeighborsClassifier as \
        BaseKNeighborsClassifier
    from sklearn.neighbors._base import _check_weights
    from sklearn.utils.validation import _deprecate_positional_args
else:
    from sklearn.neighbors.classification import KNeighborsClassifier as \
        BaseKNeighborsClassifier
    from sklearn.neighbors.base import _check_weights

    def _deprecate_positional_args(f):
        return f


def onedal_classifier_predict(estimator, X, base_predict):
    X = _check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
    onedal_model = getattr(estimator, '_onedal_model', None)
    n_features = getattr(estimator, 'n_features_in_', None)
    shape = getattr(X, 'shape', None)
    if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        raise ValueError((f'X has {X.shape[1]} features, '
                          f'but KNNClassifier is expecting '
                          f'{n_features} features as input'))

    # try:
    #     fptype = getFPType(X)
    # except ValueError:
    #     fptype = None

    # if daal_model is not None and fptype is not None and not sp.issparse(X):
    if onedal_model is not None and not sp.issparse(X):
        logging.info(
            "sklearn.neighbors.KNeighborsClassifier"
            ".predict: " + get_patch_message("onedal"))

        method = parse_auto_method(
            estimator, estimator.algorithm,
            estimator.n_samples_fit_, n_features)
        estimator._fit_method = method

        class_count = 0 if estimator.classes_ is None else len(estimator.classes_)
        params = {
            'fptype': 'float' if X.dtype is np.dtype('float32') else 'double',
            'vote_weights': 'uniform' if estimator.weights == 'uniform' else 'distance',
            'method': estimator._fit_method,
            'radius': estimator.radius,
            'class_count': class_count,
            'neighbor_count': estimator.n_neighbors,
            'leaf_size': estimator.leaf_size,
            'metric': estimator.metric,
            'p': estimator.p,
            'metric_params': estimator.metric_params,
            'n_jobs': estimator.n_jobs,
            'result_option': 'responses',
        }

        prediction_result = _onedal_predict(estimator, onedal_model, X, params)
        print(prediction_result.responses)
        responses = _get_responses(prediction_result)
        # indices = backend.to_numpy(prediction_results.indices)
        result = estimator.classes_.take(
            np.asarray(responses.ravel(), dtype=np.intp))
    else:
        logging.info(
            "sklearn.neighbors.KNeighborsClassifier"
            ".predict: " + get_patch_message("sklearn"))
        result = base_predict(estimator, X)

    return result


if LooseVersion(sklearn_version) >= LooseVersion("0.24"):
    class KNeighborsClassifier_(onedal_KNeighborsMixin, BaseClassifierMixin, onedal_NeighborsBase):
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
            self.weights = \
                weights if LooseVersion(sklearn_version) >= LooseVersion("1.0") else _check_weights(weights)
elif LooseVersion(sklearn_version) >= LooseVersion("0.22"):
    from sklearn.neighbors._base import SupervisedIntegerMixin as \
        BaseSupervisedIntegerMixin

    class KNeighborsClassifier_(onedal_NeighborsBase, onedal_KNeighborsMixin,
                                BaseSupervisedIntegerMixin, BaseClassifierMixin):
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
else:
    from sklearn.neighbors.base import SupervisedIntegerMixin as \
        BaseSupervisedIntegerMixin

    class KNeighborsClassifier_(onedal_NeighborsBase, onedal_KNeighborsMixin,
                                BaseSupervisedIntegerMixin, BaseClassifierMixin):
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


class KNeighborsClassifier(KNeighborsClassifier_):
    @_deprecate_positional_args
    def __init__(self, n_neighbors=5, *,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params,
            n_jobs=n_jobs, **kwargs)

    def fit(self, X, y):
        X_incorrect_type = isinstance(
            X, (KDTree, BallTree, onedal_NeighborsBase, BaseNeighborsBase))

        if not X_incorrect_type and self.weights in ['uniform', 'distance'] \
            and self.algorithm in ['brute', 'kd_tree', 'auto', 'ball_tree'] \
            and self.metric in ['minkowski', 'euclidean', 'chebyshev', 'cosine']:
            try:
                logging.info(
                    "sklearn.neighbors.KNeighborsClassifier."
                    "fit: " + get_patch_message("onedal"))
                result = onedal_NeighborsBase._fit(self, X, y)
            except RuntimeError:
                logging.info(
                    "sklearn.neighbors.KNeighborsClassifier."
                    "fit: " + get_patch_message("sklearn_after_onedal"))
                result = BaseNeighborsBase.fit(self, X, y)
        else:
            logging.info(
                "sklearn.neighbors.KNeighborsClassifier."
                "fit: " + get_patch_message("sklearn"))
            result = BaseNeighborsBase.fit(self, X, y)
        return result

    def predict(self, X):
        return onedal_classifier_predict(self, X, BaseKNeighborsClassifier.predict)

    def predict_proba(self, X):
        return BaseKNeighborsClassifier.predict_proba(self, X)
