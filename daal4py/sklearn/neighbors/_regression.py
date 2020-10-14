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

# daal4py KNN regression scikit-learn-compatible classes

from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion


SKLEARN_24 = LooseVersion(sklearn_version) >= LooseVersion("0.24")
SKLEARN_22 = LooseVersion(sklearn_version) >= LooseVersion("0.22")


if SKLEARN_22:
    from sklearn.neighbors._regression import KNeighborsRegressor as BaseKNeighborsRegressor
else:
    from sklearn.neighbors.regression import KNeighborsRegressor as BaseKNeighborsRegressor


class KNeighborsRegressor(BaseKNeighborsRegressor):
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        daal_model = getattr(self, '_daal_model', None)
        if daal_model is not None \
        or getattr(self, '_tree', 0) is None and self._fit_method == 'kd_tree':
            if SKLEARN_24:
                BaseKNeighborsRegressor.fit(self, self._fit_X, self._y)
            else:
                BaseKNeighborsRegressor.fit(self, self._fit_X)
        return BaseKNeighborsRegressor.kneighbors(self, X, n_neighbors, return_distance)
