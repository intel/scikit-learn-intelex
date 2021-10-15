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

# from daal4py.sklearn.neighbors import KNeighborsRegressor

from ._base import NeighborsBase, KNeighborsMixin
from onedal.neighbors import NeighborsBase as onedal_NeighborsBase
from onedal.neighbors import KNeighborsMixin as onedal_KNeighborsMixin
from sklearn.base import RegressorMixin
from .._utils import sklearn_check_version
from sklearn.neighbors._base import NeighborsBase as BaseNeighborsBase

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
    class KNeighborsRegressor_(onedal_KNeighborsMixin, RegressorMixin, onedal_NeighborsBase):
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

    class KNeighborsRegressor_(onedal_NeighborsBase, onedal_KNeighborsMixin,
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
    
    # def fit(self, X, y):
    #     return NeighborsBase._fit(self, X, y)

    def predict(self, X):
        return BaseKNeighborsRegressor.predict(self, X)
