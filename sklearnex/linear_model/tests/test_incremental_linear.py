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

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex._device_offload import _transfer_to_host
from sklearnex.linear_model import IncrementalLinearRegression


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_gold_data(dataframe, queue, fit_intercept, macro_block, dtype):
    X = np.array([[1], [2]])
    X = X.astype(dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = np.array([1, 2])
    y = y.astype(dtype=dtype)
    y_df = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    inclin = IncrementalLinearRegression(fit_intercept=fit_intercept)
    if macro_block is not None:
        hparams = inclin.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    inclin.fit(X_df, y_df)

    y_pred = inclin.predict(X_df)
    # TODO Implement more clear way to transfer dpctl/dpnp arrays to host
    _, hostdata = _transfer_to_host(queue, y_pred)
    y_pred = hostdata[0]

    assert_allclose(inclin.coef_, [1], rtol=1e-6, atol=1e-6)
    if fit_intercept:
        assert_allclose(inclin.intercept_, [0], atol=2e-6)
    assert_allclose(y_pred, y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_gold_data(
    dataframe, queue, fit_intercept, macro_block, dtype
):
    X = np.array([[1], [2], [3], [4]])
    X = X.astype(dtype=dtype)
    y = X + 3
    y = y.astype(dtype=dtype)
    X_split = np.array_split(X, 2)
    y_split = np.array_split(y, 2)

    inclin = IncrementalLinearRegression()
    if macro_block is not None:
        hparams = inclin.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        y_split_df = _convert_to_dataframe(
            y_split[i], sycl_queue=queue, target_df=dataframe
        )
        inclin.partial_fit(X_split_df, y_split_df)

    assert inclin.n_features_in_ == 1
    assert_allclose(inclin.coef_, [[1]], rtol=1e-6, atol=1e-6)
    if fit_intercept:
        assert_allclose(inclin.intercept_, 3, rtol=1e-6, atol=1e-6)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_pred = inclin.predict(X_df)
    # TODO Implement more clear way to transfer dpctl/dpnp arrays to host
    _, hostdata = _transfer_to_host(queue, y_pred)
    y_pred = hostdata[0]

    assert_allclose(y_pred, y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_multitarget_on_gold_data(
    dataframe, queue, fit_intercept, macro_block, dtype
):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    X = X.astype(dtype=dtype)
    y = np.dot(X, [1, 2]) + 3
    y = y.astype(dtype=dtype)
    X_split = np.array_split(X, 2)
    y_split = np.array_split(y, 2)

    inclin = IncrementalLinearRegression()
    if macro_block is not None:
        hparams = inclin.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    for i in range(2):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        y_split_df = _convert_to_dataframe(
            y_split[i], sycl_queue=queue, target_df=dataframe
        )
        inclin.partial_fit(X_split_df, y_split_df)

    assert inclin.n_features_in_ == 2
    assert_allclose(inclin.coef_, [1.0, 2.0], rtol=1e-5, atol=1e-5)
    if fit_intercept:
        assert_allclose(inclin.intercept_, 3.0, rtol=1e-5, atol=1e-5)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y_pred = inclin.predict(X_df)
    # TODO Implement more clear way to transfer dpctl/dpnp arrays to host
    _, hostdata = _transfer_to_host(queue, y_pred)
    y_pred = hostdata[0]
    assert_allclose(y_pred, y, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("num_samples", [100, 1000])
@pytest.mark.parametrize("num_features", [5, 10])
@pytest.mark.parametrize("num_targets", [1, 2])
@pytest.mark.parametrize("num_blocks", [1, 10])
@pytest.mark.parametrize("macro_block", [None, 1024])
@pytest.mark.parametrize("dtype", [np.float32])
def test_sklearnex_partial_fit_on_random_data(
    dataframe,
    queue,
    num_samples,
    num_features,
    num_targets,
    num_blocks,
    macro_block,
    dtype,
):
    seed = 42
    gen = np.random.default_rng(seed)
    intercept = gen.random(size=num_targets, dtype=dtype)
    coef = gen.random(size=(num_targets, num_features), dtype=dtype).T

    X = gen.random(size=(num_samples, num_features), dtype=dtype)
    y = X @ coef + intercept[np.newaxis, :]
    X_split = np.array_split(X, num_blocks)
    y_split = np.array_split(y, num_blocks)

    inclin = IncrementalLinearRegression(fit_intercept=True)
    if macro_block is not None:
        hparams = inclin.get_hyperparameters("fit")
        hparams.cpu_macro_block = macro_block
        hparams.gpu_macro_block = macro_block
    for i in range(num_blocks):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        y_split_df = _convert_to_dataframe(
            y_split[i], sycl_queue=queue, target_df=dataframe
        )
        inclin.partial_fit(X_split_df, y_split_df)

    if queue is not None and queue.sycl_device.is_gpu:
        tol = 5e-3 if inclin.coef_.dtype == np.float32 else 1e-5
    else:
        tol = 2e-3 if inclin.coef_.dtype == np.float32 else 1e-5
    assert_allclose(coef, inclin.coef_.T, rtol=tol)

    tol = 2e-3 if inclin.intercept_.dtype == np.float32 else 1e-5
    assert_allclose(intercept, inclin.intercept_, rtol=tol)

    X_test = gen.random(size=(num_samples, num_features), dtype=dtype)
    gtr = X_test @ coef + intercept[np.newaxis, :]
    X_test_df = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    # TODO Implement more clear way to transfer dpctl/dpnp arrays to host
    y_pred = inclin.predict(X_test_df)
    _, hostdata = _transfer_to_host(queue, y_pred)
    y_pred = hostdata[0]

    tol = 2e-4 if y_pred.dtype == np.float32 else 1e-7
    assert_allclose(gtr, y_pred, rtol=tol)
