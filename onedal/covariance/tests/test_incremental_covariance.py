# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_on_gold_data_unbiased(queue, dtype):
    from onedal.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X_split = np.array_split(X, 2)
    X = X.astype(dtype)
    inccov = IncrementalEmpiricalCovariance()

    for i in range(2):
        inccov.partial_fit(X_split[i], queue=queue)
    result = inccov.finalize_fit()

    expected_covariance = np.array([[0, 0], [0, 0]])
    expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[1, 2], [3, 6]])
    X_split = np.array_split(X, 2)
    X = X.astype(dtype)
    inccov = IncrementalEmpiricalCovariance()

    for i in range(2):
        inccov.partial_fit(X_split[i], queue=queue)
    result = inccov.finalize_fit()

    expected_covariance = np.array([[2, 4], [4, 8]])
    expected_means = np.array([2, 4])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_on_gold_data_biased(queue, dtype):
    from onedal.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance(bias=True)

    for i in range(2):
        inccov.partial_fit(X_split[i], queue=queue)
    result = inccov.finalize_fit()

    expected_covariance = np.array([[0, 0], [0, 0]])
    expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[1, 2], [3, 6]])
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance(bias=True)

    for i in range(2):
        inccov.partial_fit(X_split[i], queue=queue)
    result = inccov.finalize_fit()

    expected_covariance = np.array([[1, 2], [2, 4]])
    expected_means = np.array([2, 4])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("num_batches", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("row_count", [100, 1000, 2000])
@pytest.mark.parametrize("column_count", [10, 100, 200])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_partial_fit_on_random_data(
    queue, num_batches, row_count, column_count, bias, dtype
):
    from onedal.covariance import IncrementalEmpiricalCovariance

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype)
    X_split = np.array_split(X, num_batches)
    inccov = IncrementalEmpiricalCovariance(bias=bias)

    for i in range(num_batches):
        inccov.partial_fit(X_split[i], queue=queue)
    result = inccov.finalize_fit()

    expected_covariance = np.cov(X.T, bias=bias)
    expected_means = np.mean(X, axis=0)

    assert_allclose(expected_covariance, result.covariance_, atol=1e-6)
    assert_allclose(expected_means, result.location_, atol=1e-6)
