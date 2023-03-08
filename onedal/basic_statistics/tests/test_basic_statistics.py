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

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

if daal_check_version((2023, 'P', 100)):
    import pytest
    import numpy as np
    from numpy.testing import assert_allclose

    from onedal.basic_statistics import BasicStatistics
    from onedal.tests.utils._device_selection import get_queues

    options_and_tests = [
        ("sum", np.sum, (1e-5, 1e-7)),
        ("min", np.min, (1e-5, 1e-7)),
        ("max", np.max, (1e-5, 1e-7)),
        ("mean", np.mean, (1e-5, 1e-7)),
        ("standard_deviation", np.std, (3e-5, 3e-5))
    ]

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_basic_uniform(queue, dtype):
        seed = 42
        s_count, f_count = 70000, 29

        gen = np.random.default_rng(seed)
        data = gen.uniform(low=-0.5, high=+0.6,
                           size=(s_count, f_count))
        data = data.astype(dtype=dtype)

        alg = BasicStatistics(result_options="mean")
        res = alg.compute(data, queue=queue)

        res_mean = res["mean"]
        gtr_mean = np.mean(data, axis=0)
        tol = 1e-5 if res_mean.dtype == np.float32 else 1e-7
        assert_allclose(gtr_mean, res_mean, rtol=tol)

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('option', options_and_tests)
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_option_uniform(queue, option, dtype):
        seed = 77
        s_count, f_count = 19999, 31

        result_option, function, tols = option
        fp32tol, fp64tol = tols

        gen = np.random.default_rng(seed)
        data = gen.uniform(low=-0.3, high=+0.7,
                           size=(s_count, f_count))
        data = data.astype(dtype=dtype)

        alg = BasicStatistics(result_options=result_option)
        res = alg.compute(data, queue=queue)

        res, gtr = res[result_option], function(data, axis=0)

        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, rtol=tol)

    @pytest.mark.parametrize('queue', get_queues())
    @pytest.mark.parametrize('option', options_and_tests)
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    def test_option_weighted(queue, option, dtype):
        seed = 999
        s_count, f_count = 1024, 127

        result_option, function, tols = option
        fp32tol, fp64tol = tols
        fp32tol, fp64tol = 30 * fp32tol, 50 * fp64tol

        gen = np.random.default_rng(seed)
        data = gen.uniform(low=-5.0, high=+9.0,
                           size=(s_count, f_count))
        weights = gen.uniform(low=-0.5, high=+1.0,
                              size=s_count)

        data = data.astype(dtype=dtype)
        weights = weights.astype(dtype=dtype)

        alg = BasicStatistics(result_options=result_option)
        res = alg.compute(data, weights, queue=queue)

        weighted = np.diag(weights) @ data
        res, gtr = res[result_option], function(weighted, axis=0)

        tol = fp32tol if res.dtype == np.float32 else fp64tol
        assert_allclose(gtr, res, rtol=tol)
