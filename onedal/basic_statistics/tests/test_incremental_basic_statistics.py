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

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2023, "P", 100)):
    import numpy as np
    import pytest
    from numpy.testing import assert_allclose

    from onedal.basic_statistics import IncrementalBasicStatistics
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
        ("sum", expected_sum, (1e-5, 1e-7)),
        ("min", expected_min, (1e-5, 1e-7)),
        ("max", expected_max, (1e-5, 1e-7)),
        ("mean", expected_mean, (1e-5, 1e-7)),
        ("variance", expected_variance, (2e-3, 2e-2)),
        ("variation", expected_variation, (2e-2, 2e-2)),
        ("sum_squares", expected_sum_squares, (1e-5, 1e-7)),
        ("sum_squares_centered", expected_sum_squares_centered, (1e-5, 1e-7)),
        ("standard_deviation", expected_standard_deviation, (2e-3, 6e-3)),
        ("second_order_raw_moment", expected_second_order_raw_moment, (1e-5, 1e-7))
    ]
    

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("num_batches", [2, 10])
    @pytest.mark.parametrize("option", options_and_tests)
    @pytest.mark.parametrize("row_count", [100, 1000])
    @pytest.mark.parametrize("column_count", [10, 100])
    @pytest.mark.parametrize("weighted", [True, False])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_single_option(queue, num_batches, option, row_count, column_count, weighted, dtype):
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
        assert_allclose(gtr, res, rtol=tol, atol=tol)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("num_batches", [2, 10])
    @pytest.mark.parametrize("row_count", [100, 1000])
    @pytest.mark.parametrize("column_count", [10, 100])
    @pytest.mark.parametrize("weighted", [True, False])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_multiple_options(queue, num_batches, row_count, column_count, weighted, dtype):
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

        tol = 1e-5 if res_mean.dtype == np.float32 else 1e-7
        assert_allclose(gtr_mean, res_mean, rtol=tol, atol=tol)
        assert_allclose(gtr_max, res_max, rtol=tol, atol=tol)
        assert_allclose(gtr_sum, res_sum, rtol=tol, atol=tol)


    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("num_batches", [2, 10])
    @pytest.mark.parametrize("row_count", [100, 1000])
    @pytest.mark.parametrize("column_count", [10, 100])
    @pytest.mark.parametrize("weighted", [True, False])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_all_option(queue, num_batches, row_count, column_count, weighted, dtype):
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
            print(result_option)
            fp32tol, fp64tol = tols
            res = getattr(result, result_option)
            if weighted:
                gtr = function(weighted_data)
            else:
                gtr = function(data)
            tol = fp32tol if res.dtype == np.float32 else fp64tol
            assert_allclose(gtr, res, rtol=tol, atol=tol)

