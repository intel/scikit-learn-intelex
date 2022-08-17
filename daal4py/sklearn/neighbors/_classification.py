#===============================================================================
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
#===============================================================================

# daal4py KNN classification scikit-learn-compatible classes

from ._base import NeighborsBase, KNeighborsMixin
from ._base import parse_auto_method, prediction_algorithm
from sklearn.base import ClassifierMixin as BaseClassifierMixin
from .._utils import (
    getFPType,
    sklearn_check_version,
    PatchingConditionsChain)
from .._device_offload import support_usm_ndarray
from sklearn.utils.validation import check_array
import numpy as np
from scipy import sparse as sp

if sklearn_check_version("0.22"):
    from sklearn.neighbors._classification import KNeighborsClassifier as \
        BaseKNeighborsClassifier
    if not sklearn_check_version("1.2"):
        from sklearn.neighbors._base import _check_weights
    from sklearn.utils.validation import _deprecate_positional_args
else:
    from sklearn.neighbors.classification import KNeighborsClassifier as \
        BaseKNeighborsClassifier
    from sklearn.neighbors.base import _check_weights

    def _deprecate_positional_args(f):
        return f


def daal4py_classifier_predict(estimator, X, base_predict):
    if sklearn_check_version('1.0'):
        estimator._check_feature_names(X, reset=False)
    X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32])
    daal_model = getattr(estimator, '_daal_model', None)
    n_features = getattr(estimator, 'n_features_in_', None)
    shape = getattr(X, 'shape', None)
    if n_features and shape and len(shape) > 1 and shape[1] != n_features:
        raise ValueError((f'X has {X.shape[1]} features, '
                          f'but KNNClassifier is expecting '
                          f'{n_features} features as input'))

    try:
        fptype = getFPType(X)
    except ValueError:
        fptype = None

    _patching_status = PatchingConditionsChain(
        "sklearn.neighbors.KNeighborsClassifier.predict")
    _dal_ready = _patching_status.and_conditions([
        (daal_model is not None, "oneDAL model was not trained."),
        (fptype is not None, "Unable to get dtype."),
        (not sp.issparse(X), "X is sparse. Sparse input is not supported.")])
    _patching_status.write_log()

    if _dal_ready:
        params = {
            'method': 'defaultDense',
            'k': estimator.n_neighbors,
            'nClasses': len(estimator.classes_),
            'voteWeights': 'voteUniform'
            if estimator.weights == 'uniform' else 'voteDistance',
            'resultsToEvaluate': 'computeClassLabels',
            'resultsToCompute': ''
        }

        method = parse_auto_method(
            estimator, estimator.algorithm, estimator.n_samples_fit_, n_features)
        predict_alg = prediction_algorithm(method, fptype, params)
        prediction_result = predict_alg.compute(X, daal_model)
        result = estimator.classes_.take(
            np.asarray(prediction_result.prediction.ravel(), dtype=np.intp))
    else:
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
            self.weights = \
                weights if sklearn_check_version("1.0") else _check_weights(weights)
elif sklearn_check_version("0.22"):
    from sklearn.neighbors._base import SupervisedIntegerMixin as \
        BaseSupervisedIntegerMixin

    class KNeighborsClassifier_(NeighborsBase, KNeighborsMixin,
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

    class KNeighborsClassifier_(NeighborsBase, KNeighborsMixin,
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
    __doc__ = BaseKNeighborsClassifier.__doc__

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

    @support_usm_ndarray()
    def fit(self, X, y):
        """
        Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : {array-like, sparse matrix} of shape (n_samples,) or \
                (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        return NeighborsBase._fit(self, X, y)

    @support_usm_ndarray()
    def predict(self, X):
        """
        Predict the class labels for the provided data.

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
        return daal4py_classifier_predict(self, X, BaseKNeighborsClassifier.predict)

    @support_usm_ndarray()
    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of n_outputs \
                of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        if sklearn_check_version('1.0'):
            self._check_feature_names(X, reset=False)
        return BaseKNeighborsClassifier.predict_proba(self, X)
