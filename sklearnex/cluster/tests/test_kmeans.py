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

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):

    from sklearnex.cluster import KMeans

    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    assert "daal4py" in kmeans.__module__

    X_test = [[0, 0], [12, 3]]
    X_test = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)
    result = kmeans.predict(X_test)
    expected = np.array([1, 0], dtype=np.int32)
    assert_allclose(expected, _as_numpy(result))
