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

# daal4py KNN scikit-learn-compatible classes

from ._base import NeighborsBase, KNeighborsMixin, RadiusNeighborsMixin
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion


SKLEARN_23 = LooseVersion(sklearn_version) >= LooseVersion("0.23")
SKLEARN_22 = LooseVersion(sklearn_version) >= LooseVersion("0.22")
SKLEARN_21 = LooseVersion(sklearn_version) >= LooseVersion("0.21")


if SKLEARN_22:
    from sklearn.utils.validation import _deprecate_positional_args
else:
    def _deprecate_positional_args(f):
        return f


if SKLEARN_21 and not SKLEARN_22:
    class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
        def __init__(self, n_neighbors=5, radius=1.0,
                 algorithm='auto', leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=None, **kwargs):
            super().__init__(
              n_neighbors=n_neighbors,
              radius=radius,
              algorithm=algorithm,
              leaf_size=leaf_size, metric=metric, p=p,
              metric_params=metric_params, n_jobs=n_jobs, **kwargs)

        def fit(self, X, y=None):
            return NeighborsBase._fit(self, X)
elif SKLEARN_22 and not SKLEARN_23:
    class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
        def __init__(self, n_neighbors=5, radius=1.0,
                     algorithm='auto', leaf_size=30, metric='minkowski',
                     p=2, metric_params=None, n_jobs=None):
            super().__init__(
                  n_neighbors=n_neighbors,
                  radius=radius,
                  algorithm=algorithm,
                  leaf_size=leaf_size, metric=metric, p=p,
                  metric_params=metric_params, n_jobs=n_jobs)

        def fit(self, X, y=None):
            return NeighborsBase._fit(self, X)
else:
    class NearestNeighbors(KNeighborsMixin, RadiusNeighborsMixin, NeighborsBase):
        @_deprecate_positional_args
        def __init__(self, *, n_neighbors=5, radius=1.0,
                     algorithm='auto', leaf_size=30, metric='minkowski',
                     p=2, metric_params=None, n_jobs=None):
            super().__init__(
                  n_neighbors=n_neighbors,
                  radius=radius,
                  algorithm=algorithm,
                  leaf_size=leaf_size, metric=metric, p=p,
                  metric_params=metric_params, n_jobs=n_jobs)

        def fit(self, X, y=None):
            return NeighborsBase._fit(self, X)
