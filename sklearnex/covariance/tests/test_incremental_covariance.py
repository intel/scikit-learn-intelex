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
from sklearn.covariance.tests.test_covariance import (
    test_covariance,
    test_EmpiricalCovariance_validates_mahalanobis,
)

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_sklearnex_partial_fit_on_gold_data(dataframe, queue, dtype, assume_centered):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = inccov.partial_fit(X_split_df)

    if assume_centered:
        expected_covariance = np.array([[0, 0], [0, 1]])
        expected_means = np.array([0, 0])
    else:
        expected_covariance = np.array([[0, 0], [0, 0]])
        expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[1, 2], [3, 6]])
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = inccov.partial_fit(X_split_df)

    if assume_centered:
        expected_covariance = np.array([[5, 10], [10, 20]])
        expected_means = np.array([0, 0])
    else:
        expected_covariance = np.array([[1, 2], [2, 4]])
        expected_means = np.array([2, 4])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_gold_data(dataframe, queue, batch_size, dtype):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1, 2, 3], [0, -1, -2, -3], [0, 1, 2, 3], [0, 1, 2, 3]])
    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    inccov = IncrementalEmpiricalCovariance(batch_size=batch_size)

    result = inccov.fit(X_df)

    expected_covariance = np.array(
        [[0, 0, 0, 0], [0, 0.75, 1.5, 2.25], [0, 1.5, 3, 4.5], [0, 2.25, 4.5, 6.75]]
    )
    expected_means = np.array([0, 0.5, 1, 1.5])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("row_count", [10, 100])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_sklearnex_partial_fit_on_random_data(
    dataframe, queue, num_batches, row_count, column_count, dtype
):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype)
    X_split = np.array_split(X, num_batches)
    inccov = IncrementalEmpiricalCovariance(assume_centered=assume_centered)

    for i in range(num_batches):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        result = inccov.partial_fit(X_split_df)

    if assume_centered:
        expected_covariance = np.dot(X.T, X) / X.shape[0]
        expected_means = np.zeros_like(X[:,0])
    else:
        expected_covariance = np.cov(X.T, bias=1)
        expected_means = np.mean(X, axis=0)

    assert_allclose(expected_covariance, result.covariance_, atol=1e-6)
    assert_allclose(expected_means, result.location_, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("num_batches", [2, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_random_data(
    dataframe, queue, num_batches, row_count, column_count, dtype
):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    batch_size = row_count // num_batches
    inccov = IncrementalEmpiricalCovariance(batch_size=batch_size)

    result = inccov.fit(X_df)

    expected_covariance = np.cov(X.T, bias=1)
    expected_means = np.mean(X, axis=0)

    assert_allclose(expected_covariance, result.covariance_, atol=1e-6)
    assert_allclose(expected_means, result.location_, atol=1e-6)


# Monkeypatch IncrementalEmpiricalCovariance into relevant sklearn.covariance tests
@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize(
    "sklearn_test",
    [
        test_covariance,
        test_EmpiricalCovariance_validates_mahalanobis,
    ],
)
def test_IncrementalEmpiricalCovariance_against_sklearn(monkeypatch, sklearn_test):
    from sklearnex.covariance import IncrementalEmpiricalCovariance

    class_name = ".".join([sklearn_test.__module__, "EmpiricalCovariance"])
    monkeypatch.setattr(class_name, IncrementalEmpiricalCovariance)
    sklearn_test()
