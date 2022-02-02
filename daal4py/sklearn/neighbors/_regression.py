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

# daal4py KNN regression scikit-learn-compatible classes

from ._base import NeighborsBase, KNeighborsMixin
from sklearn.base import RegressorMixin
from .._utils import sklearn_check_version
from .._device_offload import support_usm_ndarray


if sklearn_check_version("0.22"):
    from sklearn.neighbors._regression import KNeighborsRegressor as \
        BaseKNeighborsRegressor
    from sklearn.neighbors._base import _check_weights
    from sklearn.utils.validation import _deprecate_positional_args
else:
    from sklearn.neighbors.regression import KNeighborsRegressor as \
        BaseKNeighborsRegressor
    from sklearn.neighbors.base import _check_weights

    def _deprecate_positional_args(f):
        return f


if sklearn_check_version("0.24"):
    class KNeighborsRegressor_(KNeighborsMixin, RegressorMixin, NeighborsBase):
        @_deprecate_positional_args
        def __init__(self, n_neighbors=5, *, weights='uniform',
                     algorithm='auto', leaf_size=30,
                     p=2, metric='minkowski', metric_params=None, n_jobs=None,
                     **kwargs):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params, n_jobs=n_jobs, **kwargs)
else:
    if sklearn_check_version("0.22"):
        from sklearn.neighbors._base import SupervisedFloatMixin as \
            BaseSupervisedFloatMixin
    else:
        from sklearn.neighbors.base import SupervisedFloatMixin as \
            BaseSupervisedFloatMixin

    class KNeighborsRegressor_(NeighborsBase, KNeighborsMixin,
                               BaseSupervisedFloatMixin, RegressorMixin):
        @_deprecate_positional_args
        def __init__(self, n_neighbors=5, *, weights='uniform',
                     algorithm='auto', leaf_size=30,
                     p=2, metric='minkowski', metric_params=None, n_jobs=None,
                     **kwargs):
            super().__init__(
                n_neighbors=n_neighbors,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params, n_jobs=n_jobs, **kwargs)


class KNeighborsRegressor(KNeighborsRegressor_):
    __doc__ = BaseKNeighborsRegressor.__doc__

    @_deprecate_positional_args
    def __init__(self, n_neighbors=5, *, weights='uniform',
                 algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self.weights = \
            weights if sklearn_check_version("1.0") else _check_weights(weights)

    def _more_tags(self):
        return BaseKNeighborsRegressor._more_tags(self)

    @support_usm_ndarray()
    def fit(self, X, y):
        """
        Fit the k-nearest neighbors regressor from the training dataset.

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
        self : KNeighborsRegressor
            The fitted k-nearest neighbors regressor.
        """
        return NeighborsBase._fit(self, X, y)

    @support_usm_ndarray()
    def predict(self, X):
        """
        Predict the target for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
            Target values.
        """
        if sklearn_check_version('1.0'):
            self._check_feature_names(X, reset=False)
        return BaseKNeighborsRegressor.predict(self, X)
