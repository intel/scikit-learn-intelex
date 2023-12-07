#!/usr/bin/env python
# ===============================================================================
# Copyright 2021 Intel Corporation
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
from numpy.testing import assert_allclose, assert_raises
from sklearn.datasets import make_regression

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.tests.utils._device_selection import is_dpctl_available


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_linear(dataframe, queue):
    from sklearnex.linear_model import LinearRegression

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    linreg = LinearRegression().fit(X, y)
    if daal_check_version((2023, "P", 100)):
        assert hasattr(linreg, "_onedal_estimator")
    assert "sklearnex" in linreg.__module__
    assert linreg.n_features_in_ == 2
    assert_allclose(_as_numpy(linreg.intercept_), 3.0)
    assert_allclose(_as_numpy(linreg.coef_), [1.0, 2.0])


def test_sklearnex_import_ridge():
    from sklearnex.linear_model import Ridge

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    ridgereg = Ridge().fit(X, y)
    assert "daal4py" in ridgereg.__module__
    assert_allclose(ridgereg.intercept_, 4.5)
    assert_allclose(ridgereg.coef_, [0.8, 1.4])


def test_sklearnex_import_lasso():
    from sklearnex.linear_model import Lasso

    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    lasso = Lasso(alpha=0.1).fit(X, y)
    assert "daal4py" in lasso.__module__
    assert_allclose(lasso.intercept_, 0.15)
    assert_allclose(lasso.coef_, [0.85, 0.0])


def test_sklearnex_import_elastic():
    from sklearnex.linear_model import ElasticNet

    X, y = make_regression(n_features=2, random_state=0)
    elasticnet = ElasticNet(random_state=0).fit(X, y)
    assert "daal4py" in elasticnet.__module__
    assert_allclose(elasticnet.intercept_, 1.451, atol=1e-3)
    assert_allclose(elasticnet.coef_, [18.838, 64.559], atol=1e-3)


@pytest.mark.skipif(
    not is_dpctl_available(["gpu"]),
    reason="gpu_blas_compute_mode is only available for select gpus",
)
def test_bf16_blas_epsilon():
    from sklearnex import config_context
    from sklearnex.linear_model import LinearRegression

    size = 100  # Needs to be under 16 bits
    X = np.ones((size, 2), dtype=np.float32)
    y = np.ones((size, 1), dtype=np.float32)
    for i in range(1, len(X)):
        X[i, 0] = X[i - 1, 0] + np.spacing(X[i - 1, 0])

    y[:, 0] = 10 * (X[:, 0] - 1)
    # Make changes in y observable in bfloat16 by first subtracting
    # to maximize fidelity (smaller spacing around 0) and multiplying
    # by 10

    # The first coeff should be under the bf16 precision and above
    # the float32 precision, meaning bf16 should yield a different
    # answer. This will prove bf16 computation in gemm on gpu has
    # been used.
    with config_context(
        target_offload="gpu:0",
        allow_fallback_to_host=False,
        gpu_blas_compute_mode="standard",
    ):
        linreg_standard = LinearRegression().fit(X, y)
    print(linreg_standard.coef_)
    print(linreg_standard.intercept_)
    with config_context(
        target_offload="gpu:0",
        allow_fallback_to_host=False,
        gpu_blas_compute_mode="float_to_bf16",
    ):
        linreg_bf16 = LinearRegression().fit(X, y)

    assert linreg_standard.n_features_in_ == 2
    assert linreg_bf16.n_features_in_ == 2
    assert_raises(
        AssertionError,
        assert_allclose,
        _as_numpy(linreg_standard.coef_),
        _as_numpy(linreg_bf16.coef_),
    )
    print(linreg_bf16.coef_)
    print(linreg_bf16.intercept_)
    assert False
