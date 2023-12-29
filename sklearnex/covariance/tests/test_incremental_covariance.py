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
from numpy.testing import assert_allclose

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_incremental_covariance(dataframe, queue):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    result = IncrementalEmpiricalCovariance(batch_size=1).fit(X)
    expected_covariance = np.array([[0, 0], [0, 0]])
    expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance()
    for i in range(2):
        result = inccov.partial_fit(X_split[i])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[0, 1, 2, 3], [0, -1, -2, -3], [0, 1, 2, 3], [0, 1, 2, 3]])
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance()
    for i in range(2):
        result = inccov.partial_fit(X_split[i])

    expected_covariance = np.array(
        [[0, 0, 0, 0], [0, 0.75, 1.5, 2.25], [0, 1.5, 3, 4.5], [0, 2.25, 4.5, 6.75]]
    )
    expected_means = np.array([0, 0.5, 1, 1.5])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)
