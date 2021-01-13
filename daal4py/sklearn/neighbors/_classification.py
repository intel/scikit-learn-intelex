#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

# daal4py KNN classification scikit-learn-compatible classes

from ._base import NeighborsBase, KNeighborsMixin
from ._base import parse_auto_method, prediction_algorithm
from sklearn.base import ClassifierMixin as BaseClassifierMixin
from .._utils import getFPType, daal_check_version, sklearn_check_version, get_patch_message
from sklearn.utils.validation import check_array
import numpy as np
from scipy import sparse as sp
import logging

if sklearn_check_version("0.22"):
    from sklearn.neighbors._classification import KNeighborsClassifier as BaseKNeighborsClassifier
    from sklearn.neighbors._base import _check_weights
    from sklearn.utils.validation import _deprecate_positional_args
else:
    from sklearn.neighbors.classification import KNeighborsClassifier as BaseKNeighborsClassifier
    from sklearn.neighbors.base import _check_weights
    def _deprecate_positional_args(f):
        return f


def daal4py_classifier_predict(estimator, X, base_predict):
    X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
    daal_model = getattr(estimator, '_daal_model', None)
    n_features = getattr(estimator, 'n_features_in_', None)
    shape = getattr(X, 'shape', None)
    if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        raise ValueError('Input data shape {} is inconsistent with the trained model'.format(X.shape))

    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    if daal_model is not None and fptype is not None and not sp.issparse(X):
        logging.info("sklearn.neighbors.KNeighborsClassifier.predict: " + get_patch_message("daal"))

        params = {
            'method': 'defaultDense',
            'k': estimator.n_neighbors,
            'nClasses': len(estimator.classes_),
            'voteWeights': 'voteUniform' if estimator.weights == 'uniform' else 'voteDistance',
            'resultsToEvaluate': 'computeClassLabels',
            'resultsToCompute': ''
        }

        method = parse_auto_method(estimator, estimator.algorithm, estimator.n_samples_fit_, n_features)
        predict_alg = prediction_algorithm(method, fptype, params)
        prediction_result = predict_alg.compute(X, daal_model)
        result = estimator.classes_.take(np.asarray(prediction_result.prediction.ravel(), dtype=np.intp))
    else:
        logging.info("sklearn.neighbors.KNeighborsClassifier.predict: " + get_patch_message("sklearn"))
        result = base_predict(estimator, X)

    return result


if sklearn_check_version("0.24"):
    class KNeighborsClassifier_(KNeighborsMixin, BaseClassifierMixin, NeighborsBase):
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
elif sklearn_check_version("0.22"):
    from sklearn.neighbors._base import SupervisedIntegerMixin as BaseSupervisedIntegerMixin
    class KNeighborsClassifier_(NeighborsBase, KNeighborsMixin, BaseSupervisedIntegerMixin, BaseClassifierMixin):
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
    from sklearn.neighbors.base import SupervisedIntegerMixin as BaseSupervisedIntegerMixin
    class KNeighborsClassifier_(NeighborsBase, KNeighborsMixin, BaseSupervisedIntegerMixin, BaseClassifierMixin):
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
        return NeighborsBase._fit(self, X, y)

    def predict(self, X):
        return daal4py_classifier_predict(self, X, BaseKNeighborsClassifier.predict)

    def predict_proba(self, X):
        return BaseKNeighborsClassifier.predict_proba(self, X)
