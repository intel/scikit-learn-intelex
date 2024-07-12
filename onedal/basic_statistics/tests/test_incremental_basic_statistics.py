# ==============================================================================
# Copyright 2024 Intel Corporation
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

from onedal.basic_statistics import IncrementalBasicStatistics
from onedal.basic_statistics.tests.test_basic_statistics import (
    expected_max,
    expected_mean,
    expected_sum,
    options_and_tests,
)
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_gold_data(queue, weighted, dtype):
    X = np.array([[0, 0], [1, 1]])
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, 2)
    if weighted:
        weights = np.array([1, 0.5])
        weights = weights.astype(dtype=dtype)
        weights_split = np.array_split(weights, 2)

    incbs = IncrementalBasicStatistics()
    for i in range(2):
        if weighted:
            incbs.partial_fit(X_split[i], weights_split[i], queue=queue)
        else:
            incbs.partial_fit(X_split[i], queue=queue)

    result = incbs.finalize_fit()

    if weighted:
        expected_weighted_mean = np.array([0.25, 0.25])
        expected_weighted_min = np.array([0, 0])
        expected_weighted_max = np.array([0.5, 0.5])
        assert_allclose(expected_weighted_mean, result.mean)
        assert_allclose(expected_weighted_max, result.max)
        assert_allclose(expected_weighted_min, result.min)
    else:
        expected_mean = np.array([0.5, 0.5])
        expected_min = np.array([0, 0])
        expected_max = np.array([1, 1])
        assert_allclose(expected_mean, result.mean)
        assert_allclose(expected_max, result.max)
        assert_allclose(expected_min, result.min)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("option", options_and_tests)
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_option_on_random_data(
    queue, num_batches, option, row_count, column_count, weighted, dtype
):
    result_option, function, tols = option
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    data = data.astype(dtype=dtype)
    data_split = np.array_split(data, num_batches)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_split = np.array_split(weights, num_batches)
    incbs = IncrementalBasicStatistics(result_options=result_option)

    for i in range(num_batches):
        if weighted:
            incbs.partial_fit(data_split[i], weights_split[i], queue=queue)
        else:
            incbs.partial_fit(data_split[i], queue=queue)
    result = incbs.finalize_fit()

    res = getattr(result, result_option)
    if weighted:
        weighted_data = np.diag(weights) @ data
        gtr = function(weighted_data)
    else:
        gtr = function(data)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_random_data(
    queue, num_batches, row_count, column_count, weighted, dtype
):
    seed = 42
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    data = data.astype(dtype=dtype)
    data_split = np.array_split(data, num_batches)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_split = np.array_split(weights, num_batches)
    incbs = IncrementalBasicStatistics(result_options=["mean", "max", "sum"])

    for i in range(num_batches):
        if weighted:
            incbs.partial_fit(data_split[i], weights_split[i], queue=queue)
        else:
            incbs.partial_fit(data_split[i], queue=queue)
    result = incbs.finalize_fit()

    res_mean, res_max, res_sum = result.mean, result.max, result.sum
    if weighted:
        weighted_data = np.diag(weights) @ data
        gtr_mean, gtr_max, gtr_sum = (
            expected_mean(weighted_data),
            expected_max(weighted_data),
            expected_sum(weighted_data),
        )
    else:
        gtr_mean, gtr_max, gtr_sum = (
            expected_mean(data),
            expected_max(data),
            expected_sum(data),
        )

    tol = 3e-4 if res_mean.dtype == np.float32 else 1e-7
    assert_allclose(gtr_mean, res_mean, atol=tol)
    assert_allclose(gtr_max, res_max, atol=tol)
    assert_allclose(gtr_sum, res_sum, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_option_on_random_data(
    queue, num_batches, row_count, column_count, weighted, dtype
):
    seed = 77
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    data = data.astype(dtype=dtype)
    data_split = np.array_split(data, num_batches)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_split = np.array_split(weights, num_batches)
    incbs = IncrementalBasicStatistics(result_options="all")

    for i in range(num_batches):
        if weighted:
            incbs.partial_fit(data_split[i], weights_split[i], queue=queue)
        else:
            incbs.partial_fit(data_split[i], queue=queue)
    result = incbs.finalize_fit()

    if weighted:
        weighted_data = np.diag(weights) @ data

    for option in options_and_tests:
        result_option, function, tols = option
        fp32tol, fp64tol = tols
        res = getattr(result, result_option)
        if weighted:
            gtr = function(weighted_data)
        else:
            gtr = function(data)
        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, atol=tol)
