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
from scipy import sparse as sp

from daal4py.sklearn._utils import daal_check_version
from onedal.basic_statistics import BasicStatistics
from onedal.tests.utils._device_selection import get_queues


def expected_sum(X):
    return np.sum(X, axis=0)


def expected_max(X):
    return np.max(X, axis=0)


def expected_min(X):
    return np.min(X, axis=0)


def expected_mean(X):
    return np.mean(X, axis=0)


def expected_standard_deviation(X):
    return np.std(X, axis=0)


def expected_variance(X):
    return np.var(X, axis=0)


def expected_variation(X):
    return expected_standard_deviation(X) / expected_mean(X)


def expected_sum_squares(X):
    return np.sum(np.square(X), axis=0)


def expected_sum_squares_centered(X):
    return np.sum(np.square(X - expected_mean(X)), axis=0)


def expected_standard_deviation(X):
    return np.sqrt(expected_variance(X))


def expected_second_order_raw_moment(X):
    return np.mean(np.square(X), axis=0)


options_and_tests = [
    ("sum", expected_sum, (5e-4, 1e-7)),
    ("min", expected_min, (1e-7, 1e-7)),
    ("max", expected_max, (1e-7, 1e-7)),
    ("mean", expected_mean, (5e-7, 1e-7)),
    ("variance", expected_variance, (2e-3, 2e-3)),
    ("variation", expected_variation, (5e-2, 5e-2)),
    ("sum_squares", expected_sum_squares, (2e-4, 1e-7)),
    ("sum_squares_centered", expected_sum_squares_centered, (2e-4, 1e-7)),
    ("standard_deviation", expected_standard_deviation, (2e-3, 2e-3)),
    ("second_order_raw_moment", expected_second_order_raw_moment, (1e-6, 1e-7)),
]

options_and_tests_csr = [
    ("sum", "sum", (5e-6, 1e-9)),
    ("min", "min", (0, 0)),
    ("max", "max", (0, 0)),
    ("mean", "mean", (5e-6, 1e-9)),
]


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("option", options_and_tests)
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_option_on_random_data(
    queue, option, row_count, column_count, weighted, dtype
):
    result_option, function, tols = option
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    data = data.astype(dtype=dtype)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
    else:
        weights = None

    basicstat = BasicStatistics(result_options=result_option)

    result = basicstat.fit(data, sample_weight=weights, queue=queue)

    res = getattr(result, result_option)
    if weighted:
        weighted_data = np.diag(weights) @ data
        gtr = function(weighted_data)
    else:
        gtr = function(data)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_random_data(queue, row_count, column_count, weighted, dtype):
    seed = 42
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    data = data.astype(dtype=dtype)

    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
    else:
        weights = None

    basicstat = BasicStatistics(result_options=["mean", "max", "sum"])

    result = basicstat.fit(data, sample_weight=weights, queue=queue)

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

    tol = 5e-4 if res_mean.dtype == np.float32 else 1e-7
    assert_allclose(gtr_mean, res_mean, atol=tol)
    assert_allclose(gtr_max, res_max, atol=tol)
    assert_allclose(gtr_sum, res_sum, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_option_on_random_data(queue, row_count, column_count, weighted, dtype):
    seed = 77
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    data = data.astype(dtype=dtype)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
    else:
        weights = None

    basicstat = BasicStatistics(result_options="all")

    result = basicstat.fit(data, sample_weight=weights, queue=queue)

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


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("option", options_and_tests)
@pytest.mark.parametrize("data_size", [100, 1000])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_1d_input_on_random_data(queue, option, data_size, weighted, dtype):
    result_option, function, tols = option
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    data = gen.uniform(low=-0.3, high=+0.7, size=data_size)
    data = data.astype(dtype=dtype)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=data_size)
        weights = weights.astype(dtype=dtype)
    else:
        weights = None

    basicstat = BasicStatistics(result_options=result_option)

    result = basicstat.fit(data, sample_weight=weights, queue=queue)

    res = getattr(result, result_option)
    if weighted:
        weighted_data = weights * data
        gtr = function(weighted_data)
    else:
        gtr = function(data)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


@pytest.mark.skipif(not hasattr(sp, "random_array"), reason="requires scipy>=1.12.0")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_basic_csr(queue, dtype):
    seed = 42
    row_count, column_count = 5000, 3008

    gen = np.random.default_rng(seed)

    data = sp.random_array(
        shape=(row_count, column_count),
        density=0.01,
        format="csr",
        dtype=dtype,
        random_state=gen,
    )

    basicstat = BasicStatistics(result_options="mean")
    result = basicstat.fit(data, queue=queue)

    res_mean = result.mean
    gtr_mean = data.mean(axis=0)
    tol = 5e-6 if res_mean.dtype == np.float32 else 1e-9
    assert_allclose(gtr_mean, res_mean, rtol=tol)


@pytest.mark.skipif(not hasattr(sp, "random_array"), reason="requires scipy>=1.12.0")
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("option", options_and_tests_csr)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_options_csr(queue, option, dtype):
    result_option, function, tols = option
    fp32tol, fp64tol = tols

    if result_option == "max":
        pytest.skip("There is a bug in oneDAL's max computations on GPU")

    seed = 42
    row_count, column_count = 20046, 4007

    gen = np.random.default_rng(seed)

    data = sp.random_array(
        shape=(row_count, column_count),
        density=0.002,
        format="csr",
        dtype=dtype,
        random_state=gen,
    )

    basicstat = BasicStatistics(result_options=result_option)
    result = basicstat.fit(data, queue=queue)

    res = getattr(result, result_option)
    func = getattr(data, function)
    gtr = func(axis=0)
    if type(gtr).__name__ != "ndarray":
        gtr = gtr.toarray().flatten()
    tol = fp32tol if res.dtype == np.float32 else fp64tol

    assert_allclose(gtr, res, rtol=tol)


def test_warning():
    basicstat = BasicStatistics()
    data = np.array([0, 1])

    with pytest.warns(
        UserWarning,
        match="Method `compute` was deprecated in version 2024.7 and will be removed in 2025.0. Use `fit` instead.",
    ) as warn_record:
        basicstat.compute(data)

    if daal_check_version((2025, "P", 0)):
        assert len(warn_record) == 0
    else:
        assert len(warn_record) == 1
