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
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs

from daal4py.sklearn._utils import daal_check_version
from onedal.basic_statistics.tests.test_basic_statistics import (
    expected_max,
    expected_mean,
    expected_min,
    expected_second_order_raw_moment,
    expected_standard_deviation,
    expected_sum,
    expected_sum_squares,
    expected_sum_squares_centered,
    expected_variance,
    expected_variation,
    options_and_tests,
)
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
    get_queues,
)
from sklearnex.basic_statistics import BasicStatistics


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_basic_statistics(dataframe, queue):
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

    result = BasicStatistics().fit(X_df, sample_weight=weights_df)

    expected_weighted_mean = np.array([0.25, 0.25])
    expected_weighted_min = np.array([0, 0])
    expected_weighted_max = np.array([0.5, 0.5])

    assert_allclose(expected_weighted_mean, result.mean)
    assert_allclose(expected_weighted_min, result.min)
    assert_allclose(expected_weighted_max, result.max)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_gold_data(dataframe, queue, weighted, dtype):
    X = np.array([[0, 0], [1, 1]])
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = np.array([1, 0.5])
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics()

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

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


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("option", options_and_tests)
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_option_on_random_data(
    dataframe, queue, option, row_count, column_count, weighted, dtype
):
    result_option, function, tols = option
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options=result_option)

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    res = getattr(result, result_option)
    if weighted:
        weighted_data = np.diag(weights) @ X
        gtr = function(weighted_data)
    else:
        gtr = function(X)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multiple_options_on_random_data(
    dataframe, queue, row_count, column_count, weighted, dtype
):
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options=["mean", "max", "sum"])

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    res_mean, res_max, res_sum = result.mean, result.max, result.sum
    if weighted:
        weighted_data = np.diag(weights) @ X
        gtr_mean, gtr_max, gtr_sum = (
            expected_mean(weighted_data),
            expected_max(weighted_data),
            expected_sum(weighted_data),
        )
    else:
        gtr_mean, gtr_max, gtr_sum = (
            expected_mean(X),
            expected_max(X),
            expected_sum(X),
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
def test_multiple_options_on_random_sparse_data(
    queue, row_count, column_count, weighted, dtype
):
    seed = 77
    random_state = 42

    if weighted:
        pytest.skip("Weighted sparse computation is not supported for sparse data")
    gen = np.random.default_rng(seed)
    X, _ = make_blobs(
        n_samples=row_count, n_features=column_count, random_state=random_state
    )
    density = 0.05
    X_sparse = csr_matrix(X * (np.random.rand(*X.shape) < density))
    X_dense = X_sparse.toarray()

    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=row_count)
        weights = weights.astype(dtype=dtype)

    options = [
        "sum",
        "max",
        "min",
        "mean",
        "standard_deviation" "variance",
        "sum_squares",
        "sum_squares_centered",
        "second_order_raw_moment",
    ]
    basicstat = BasicStatistics(result_options=options)
    if result_option == "max":
        pytest.skip("There is a bug in oneDAL's max computations on GPU")

    if weighted:
        result = basicstat.fit(X_sparse, sample_weight=weights)
    else:
        result = basicstat.fit(X_sparse)

    if weighted:
        weighted_data = np.diag(weights) @ X_dense

        gtr_sum = expected_sum(weighted_data)
        gtr_min = expected_min(weighted_data)
        gtr_mean = expected_mean(weighted_data)
        gtr_std = expected_standard_deviation(weighted_data)
        gtr_var = expected_variance(weighted_data)
        gtr_variation = expected_variation(weighted_data)
        gtr_ss = expected_sum_squares(weighted_data)
        gtr_ssc = expected_sum_squares_centered(weighted_data)
        gtr_seconf_moment = expected_second_order_raw_moment(weighted_data)
    else:
        gtr_sum = expected_sum(X_dense)
        gtr_min = expected_min(X_dense)
        gtr_mean = expected_mean(X_dense)
        gtr_std = expected_standard_deviation(X_dense)
        gtr_var = expected_variance(X_dense)
        gtr_variation = expected_variation(X_dense)
        gtr_ss = expected_sum_squares(X_dense)
        gtr_ssc = expected_sum_squares_centered(X_dense)
        gtr_seconf_moment = expected_second_order_raw_moment(X_dense)

    tol = 5e-4 if res_mean.dtype == np.float32 else 1e-7
    assert_allclose(gtr_sum, result.sum_, atol=tol)
    assert_allclose(gtr_min, result.min_, atol=tol)
    assert_allclose(gtr_mean, result.mean_, atol=tol)
    assert_allclose(gtr_std, result.standard_deviation_, atol=tol)
    assert_allclose(gtr_var, result.variance_, atol=tol)
    assert_allclose(gtr_variation, result.variation_, atol=tol)
    assert_allclose(gtr_ss, result.sum_squares_, atol=tol)
    assert_allclose(gtr_ssc, result.sum_squares_centered_, atol=tol)
    assert_allclose(gtr_seconf_moment, result.second_order_raw_moment_, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_option_on_random_data(
    dataframe, queue, row_count, column_count, weighted, dtype
):
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=+1.0, size=row_count)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options="all")

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    if weighted:
        weighted_data = np.diag(weights) @ X

    for option in options_and_tests:
        result_option, function, tols = option
        fp32tol, fp64tol = tols
        res = getattr(result, result_option)
        if weighted:
            gtr = function(weighted_data)
        else:
            gtr = function(X)
        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("option", options_and_tests)
@pytest.mark.parametrize("data_size", [100, 1000])
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_1d_input_on_random_data(dataframe, queue, option, data_size, weighted, dtype):
    result_option, function, tols = option
    fp32tol, fp64tol = tols
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=data_size)
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    if weighted:
        weights = gen.uniform(low=-0.5, high=1.0, size=data_size)
        weights = weights.astype(dtype=dtype)
        weights_df = _convert_to_dataframe(weights, sycl_queue=queue, target_df=dataframe)
    basicstat = BasicStatistics(result_options=result_option)

    if weighted:
        result = basicstat.fit(X_df, sample_weight=weights_df)
    else:
        result = basicstat.fit(X_df)

    res = getattr(result, result_option)
    if weighted:
        weighted_data = weights * X
        gtr = function(weighted_data)
    else:
        gtr = function(X)

    tol = fp32tol if res.dtype == np.float32 else fp64tol
    assert_allclose(gtr, res, atol=tol)


def test_warning():
    basicstat = BasicStatistics("all")
    data = np.array([0, 1])

    basicstat.fit(data)
    for i in basicstat._onedal_estimator.get_all_result_options():
        with pytest.warns(
            UserWarning,
            match="Result attributes without a trailing underscore were deprecated in version 2025.1 and will be removed in 2026.0",
        ) as warn_record:
            getattr(basicstat, i)

        if daal_check_version((2026, "P", 0)):
            assert len(warn_record) == 0, i
        else:
            assert len(warn_record) == 1, i
