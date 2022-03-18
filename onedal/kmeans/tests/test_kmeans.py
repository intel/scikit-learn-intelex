#===============================================================================
# Copyright 2022 Intel Corporation
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

import pytest
import numpy as np

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

from onedal.kmeans import KMeans
from onedal.tests.utils._device_selection import get_queues

from sklearn import datasets


@pytest.mark.parametrize('queue', get_queues())
def test_iris(queue):
    iris = datasets.load_iris()
    clf = KMeans(n_clusters=2).fit(iris.data, queue=queue)

@pytest.mark.parametrize('queue', get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_results(dtype, queue):
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    sample_weight = [3, 1, 1, 3]
    init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.375
    expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=dtype)
    expected_n_iter = 2

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
    kmeans.fit(X, queue=queue)

    # assert_array_equal(kmeans.labels_, expected_labels) wrong format
    # assert_allclose(kmeans.inertia_, expected_inertia)
    # assert_allclose(kmeans.cluster_centers_, expected_centers)
    # assert kmeans.n_iter_ == expected_n_iter
