# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):
    from sklearnex.cluster import KMeans

    X_train = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    X_test = np.array([[0, 0], [12, 3]])
    X_train = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
    if daal_check_version((2023, "P", 200)):
        assert "sklearnex" in kmeans.__module__
    else:
        assert "daal4py" in kmeans.__module__

    result_cluster_labels = kmeans.predict(X_test)
    if queue and queue.sycl_device.is_gpu:
        # KMeans Init Dense GPU implementation is different from CPU
        expected_cluster_labels = np.array([0, 1], dtype=np.int32)
    else:
        expected_cluster_labels = np.array([1, 0], dtype=np.int32)
    assert_allclose(expected_cluster_labels, _as_numpy(result_cluster_labels))
