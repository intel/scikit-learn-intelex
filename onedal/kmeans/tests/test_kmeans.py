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

centers = np.array([[0., 0.], [1., 1.]])

@pytest.mark.parametrize(
    "init",
    ["random", "k-means++", centers, lambda X, k, random_state: centers],
    ids=["random", "k-means++", "ndarray", "callable"],
)
@pytest.mark.parametrize('queue', get_queues())
def test_all_init(init, queue):
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=np.float32)
    n_clusters = 2
    kmeans = KMeans(init=init, n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    assert kmeans.labels_.shape[0] == X.shape[0]

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('queue', get_queues())
def test_kmeans_fit(dtype, queue):
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    init_centers = np.array([[0., 0.], [1., 1.]], dtype=dtype)

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.25
    expected_centers = np.array([[0.25, 0.  ], [0.75, 1.  ]], dtype=dtype)
    expected_n_iter = 2

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
    kmeans.fit(X, queue=queue)

    assert_array_equal(kmeans.labels_, expected_labels)
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert_allclose(kmeans.cluster_centers_, expected_centers)
    assert kmeans.n_iter_ == expected_n_iter


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('queue', get_queues())
def test_kmeans_predict(dtype, queue):
    X = np.array([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    y = np.array([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.9]], dtype=dtype)
    init_centers = np.array([[0., 0.], [1., 1.]], dtype=dtype)
    expected_labels = [0., 0., 1., 1.]

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers)
    kmeans.fit(X, queue=queue)
    labels = kmeans.predict(y)

    assert_array_equal(labels, expected_labels)
