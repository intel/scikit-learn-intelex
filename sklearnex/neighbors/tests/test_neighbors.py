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

import numpy as np
from numpy.testing import assert_allclose


def test_sklearnex_import_knn_classifier():
    from sklearnex.neighbors import KNeighborsClassifier
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    assert 'sklearnex' in neigh.__module__
    assert_allclose(neigh.predict([[1.1]]), [0])


def test_sklearnex_import_knn_regression():
    from sklearnex.neighbors import KNeighborsRegressor
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    assert 'sklearnex' in neigh.__module__
    assert_allclose(neigh.predict([[1.5]]), [0.5])


def test_sklearnex_import_nn():
    from sklearnex.neighbors import NearestNeighbors
    X = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
    neigh = NearestNeighbors(n_neighbors=2).fit(X)
    assert 'sklearnex' in neigh.__module__
    result = neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
    assert_allclose(result, [[2, 0]])
