# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

options_and_tests = [
    ("sum", np.sum, (1e-5, 1e-7)),
    ("min", np.min, (1e-5, 1e-7)),
    ("max", np.max, (1e-5, 1e-7)),
    ("mean", np.mean, (1e-5, 1e-7)),
    ("standard_deviation", np.std, (3e-5, 3e-5)),
]


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_basic_statistics(dataframe, queue):
    from sklearnex.basic_statistics import BasicStatistics

    X = np.array([[0, 0], [1, 1]])
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    weights = np.array([1, 0.5])
    weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)

    result = BasicStatistics().fit(X_df)

    expected_mean = np.array([0.5, 0.5])
    expected_min = np.array([0, 0])
    expected_max = np.array([1, 1])

    assert_allclose(expected_mean, result.mean)
    assert_allclose(expected_max, result.max)
    assert_allclose(expected_min, result.min)

    result = BasicStatistics().fit(X_df, weights_df)

    expected_weighted_mean = np.array([0.25, 0.25])
    expected_weighted_min = np.array([0, 0])
    expected_weighted_max = np.array([0.5, 0.5])

    assert_allclose(expected_weighted_mean, result.mean)
    assert_allclose(expected_weighted_max, result.max)
    assert_allclose(expected_weighted_min, result.min)
