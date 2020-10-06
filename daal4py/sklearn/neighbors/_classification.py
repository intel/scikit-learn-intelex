#
# *******************************************************************************
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
# ******************************************************************************/

# daal4py KNN classification scikit-learn-compatible classes

from ._base import NeighborsBase, KNeighborsMixin
from sklearn.base import ClassifierMixin as BaseClassifierMixin
from sklearn.neighbors._classification import KNeighborsClassifier as BaseKNeighborsClassifier
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.neighbors._base import _check_weights


class KNeighborsClassifier(KNeighborsMixin, BaseClassifierMixin, NeighborsBase):
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

    def fit(self, X, y):
        return BaseKNeighborsClassifier.fit(self, X, y)

    def predict(self, X):
        return BaseKNeighborsClassifier.predict(self, X)

    def predict_proba(self, X):
        return BaseKNeighborsClassifier.predict_proba(self, X)
