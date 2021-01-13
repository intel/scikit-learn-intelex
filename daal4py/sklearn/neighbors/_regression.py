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

# daal4py KNN regression scikit-learn-compatible classes

from ._base import NeighborsBase, KNeighborsMixin
from sklearn.base import RegressorMixin
from sklearn.utils.deprecation import deprecated
from .._utils import sklearn_check_version


if sklearn_check_version("0.22"):
    from sklearn.neighbors._regression import KNeighborsRegressor as BaseKNeighborsRegressor
    from sklearn.neighbors._base import _check_weights
    from sklearn.utils.validation import _deprecate_positional_args
else:
    from sklearn.neighbors.regression import KNeighborsRegressor as BaseKNeighborsRegressor
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
            self.weights = _check_weights(weights)
else:
    if sklearn_check_version("0.22"):
        from sklearn.neighbors._base import SupervisedFloatMixin as BaseSupervisedFloatMixin
    else:
        from sklearn.neighbors.base import SupervisedFloatMixin as BaseSupervisedFloatMixin
    class KNeighborsRegressor_(NeighborsBase, KNeighborsMixin, BaseSupervisedFloatMixin, RegressorMixin):
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
            self.weights = _check_weights(weights)


class KNeighborsRegressor(KNeighborsRegressor_):
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
       self.weights = _check_weights(weights)

    def _more_tags(self):
        return BaseKNeighborsRegressor._more_tags(self)

    def fit(self, X, y):
        return NeighborsBase._fit(self, X, y)

    def predict(self, X):
        return BaseKNeighborsRegressor.predict(self, X)
