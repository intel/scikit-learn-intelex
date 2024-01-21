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
def test_sklearnex_import_incremental_basic_statistics(dataframe, queue):
    from sklearnex.basic_statistics import IncrementalBasicStatistics

    X = np.array([[0, 0], [1, 1]])
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    weights = np.array([1, 0.5])
    weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)

    result = IncrementalBasicStatistics(batch_size=1).fit(X_df)
    expected_mean = np.array([0.5, 0.5])
    expected_min = np.array([0, 0])
    expected_max = np.array([1, 1])

    assert_allclose(expected_mean, result.mean)
    assert_allclose(expected_max, result.max)
    assert_allclose(expected_min, result.min)

    result = IncrementalBasicStatistics(batch_size=1).fit(X_df, weights_df)
    expected_weighted_mean = np.array([0.25, 0.25])
    expected_weighted_min = np.array([0, 0])
    expected_weighted_max = np.array([0.5, 0.5])

    assert_allclose(expected_weighted_mean, result.mean)
    assert_allclose(expected_weighted_max, result.max)
    assert_allclose(expected_weighted_min, result.min)

    X_split = np.array_split(X, 2)
    incbs = IncrementalBasicStatistics()
    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = incbs.partial_fit(X_split_df)

    assert_allclose(expected_mean, result.mean)
    assert_allclose(expected_max, result.max)
    assert_allclose(expected_min, result.min)

    weights_split = np.array_split(weights, 2)
    incbs = IncrementalBasicStatistics()
    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        weights_split_df = _convert_to_dataframe(
            weights_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = incbs.partial_fit(X_split_df, weights_split_df)

    assert_allclose(expected_weighted_mean, result.mean)
    assert_allclose(expected_weighted_max, result.max)
    assert_allclose(expected_weighted_min, result.min)
