# ===============================================================================
# Copyright 2023 Intel Corporation
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
# ===============================================================================

import numpy as np
import pytest
from sklearn.cluster.tests.common import generate_clustered_data

from onedal.cluster import DBSCAN
from onedal.tests.utils._device_selection import get_queues


# TODO:
# tests will be added.
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dbscan(queue, dtype):
    eps = 0.8
    min_samples = 10
    metric = "euclidean"

    n_clusters = 3
    X = generate_clustered_data(n_clusters=n_clusters)
    X = np.asarray(X, dtype=dtype)
    db = DBSCAN(metric=metric, eps=eps, min_samples=min_samples)
    labels = db.fit(X, queue).labels_

    n_clusters_2 = len(set(labels)) - int(-1 in labels)
