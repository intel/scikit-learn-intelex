# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal import _backend
from onedal._device_offload import dpctl_available, dpnp_available
from onedal.datatypes import from_table, to_table
from onedal.primitives import linear_kernel
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.tests.utils._device_selection import get_queues
from onedal.utils._array_api import _get_sycl_namespace

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp


def _test_input_format_c_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="C")
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_contiguous_numpy(queue, dtype)


def _test_input_format_f_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="F")
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_f_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_f_contiguous_numpy(queue, dtype)


def _test_input_format_c_not_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    dummy_data = np.insert(x_default, range(1, x_default.shape[1]), 8, axis=1)
    x_numpy = dummy_data[:, ::2]

    assert_allclose(x_numpy, x_default)

    assert not x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_not_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_not_contiguous_numpy(queue, dtype)


def _test_input_format_c_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="C")
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_contiguous_pandas(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_contiguous_pandas(queue, dtype)


def _test_input_format_f_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="F")
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_f_contiguous_pandas(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_f_contiguous_pandas(queue, dtype)


def _test_conversion_to_table(dtype):
    np.random.seed()
    if dtype in [np.int32, np.int64]:
        x = np.random.randint(0, 10, (15, 3), dtype=dtype)
    else:
        x = np.random.uniform(-2, 2, (18, 6)).astype(dtype)
    x_table = to_table(x)
    x2 = from_table(x_table)
    assert x.dtype == x2.dtype
    assert np.array_equal(x, x2)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_conversion_to_table(dtype):
    _test_conversion_to_table(dtype)


def _check_attributes_for_zero_copy(original_dp_tensor, dp_tensor_from_table, order):
    # dpctl.tensor is the dpnp.ndarrays's core tensor structure along
    # with advanced device management. Convert dpnp to dpctl.tensor with zero copy
    if dpnp_available and isinstance(original_dp_tensor, dpnp.ndarray):
        # Now DPCtl tensors
        original_dp_tensor = original_dp_tensor.get_array()
        dp_tensor_from_table = dp_tensor_from_table.get_array()

    assert (
        original_dp_tensor.__sycl_usm_array_interface__["data"][0]
        == dp_tensor_from_table.__sycl_usm_array_interface__["data"][0]
    )
    assert original_dp_tensor.shape == dp_tensor_from_table.shape
    assert original_dp_tensor.strides == dp_tensor_from_table.strides
    assert original_dp_tensor.dtype == dp_tensor_from_table.dtype
    if order == "F":
        assert original_dp_tensor.flags.f_contiguous
        assert dp_tensor_from_table.flags.f_contiguous
    else:
        assert original_dp_tensor.flags.c_contiguous
        assert dp_tensor_from_table.flags.c_contiguous
    assert original_dp_tensor.flags == dp_tensor_from_table.flags
    assert original_dp_tensor.sycl_queue == dp_tensor_from_table.sycl_queue


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu, gpu")
)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_input_sua_iface_zero_copy(dataframe, queue, order, dtype):
    rng = np.random.RandomState(0)
    X_default = np.array(5 * rng.random_sample((10, 59)), dtype=dtype)

    X_numpy = np.asanyarray(X_default, dtype=dtype, order=order)

    X_dp = _convert_to_dataframe(X_numpy, sycl_queue=queue, target_df=dataframe)

    sua_iface, X_dp_namespace, _ = _get_sycl_namespace(X_dp)

    X_table = to_table(X_dp, sua_iface=sua_iface)

    assert hasattr(X_table, "__sycl_usm_array_interface__")

    X_dp_from_table = from_table(X_table, sua_iface=sua_iface, xp=X_dp_namespace)

    _check_attributes_for_zero_copy(X_dp, X_dp_from_table, order)


# TODO:
# def test_wrong_inputs_for_sua_iface_conversion
