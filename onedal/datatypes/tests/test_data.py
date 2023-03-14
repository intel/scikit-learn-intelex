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

import pytest
import numpy as np
from numpy.testing import assert_allclose
from onedal.primitives import linear_kernel

from onedal import _backend

from onedal.tests.utils._device_selection import get_queues

try:
    import dpctl
    import dpctl.tensor as dpt
    dpctl_available = dpctl.__version__ >= '0.14'
except ImportError:
    dpctl_available = False


def _test_input_format_c_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='C')
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


# TODO: investigate sporadic failures on GPU
@pytest.mark.parametrize('queue', get_queues('cpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_c_contiguous_numpy(queue, dtype):
    _test_input_format_c_contiguous_numpy(queue, dtype)


def _test_input_format_f_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='F')
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


# TODO: investigate sporadic failures on GPU
@pytest.mark.parametrize('queue', get_queues('cpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_f_contiguous_numpy(queue, dtype):
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


# TODO: investigate sporadic failures on GPU
@pytest.mark.parametrize('queue', get_queues('cpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_c_not_contiguous_numpy(queue, dtype):
    _test_input_format_c_not_contiguous_numpy(queue, dtype)


def _test_input_format_c_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='C')
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


# TODO: investigate sporadic failures on GPU
@pytest.mark.parametrize('queue', get_queues('cpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_c_contiguous_pandas(queue, dtype):
    _test_input_format_c_contiguous_pandas(queue, dtype)


def _test_input_format_f_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='F')
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


# TODO: investigate sporadic failures on GPU
@pytest.mark.parametrize('queue', get_queues('cpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_input_format_f_contiguous_pandas(queue, dtype):
    _test_input_format_f_contiguous_pandas(queue, dtype)


@pytest.mark.skipif(not dpctl_available,
                    reason="requires dpctl>=0.14")
@pytest.mark.parametrize('queue', get_queues('cpu,gpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_input_format_c_contiguous_dpctl(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 59)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='C')
    x_dpt = dpt.asarray(x_numpy, usm_type="device", sycl_queue=queue)
    # assert not x_dpt.flags.fnc
    assert isinstance(x_dpt, dpt.usm_ndarray)

    x_table = _backend.dpctl_to_table(x_dpt)
    assert hasattr(x_table, '__sycl_usm_array_interface__')
    x_dpt_from_table = dpt.asarray(x_table)

    assert x_dpt.__sycl_usm_array_interface__[
        'data'][0] == x_dpt_from_table.__sycl_usm_array_interface__['data'][0]
    assert x_dpt.shape == x_dpt_from_table.shape
    assert x_dpt.strides == x_dpt_from_table.strides
    assert x_dpt.dtype == x_dpt_from_table.dtype
    assert x_dpt.flags.c_contiguous
    assert x_dpt_from_table.flags.c_contiguous


@pytest.mark.skipif(not dpctl_available,
                    reason="requires dpctl>=0.14")
@pytest.mark.parametrize('queue', get_queues('cpu,gpu'))
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_input_format_f_contiguous_dpctl(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 59)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order='F')
    x_dpt = dpt.asarray(x_numpy, usm_type="device", sycl_queue=queue)
    # assert not x_dpt.flags.fnc
    assert isinstance(x_dpt, dpt.usm_ndarray)

    x_table = _backend.dpctl_to_table(x_dpt)
    assert hasattr(x_table, '__sycl_usm_array_interface__')
    x_dpt_from_table = dpt.asarray(x_table)

    assert x_dpt.__sycl_usm_array_interface__[
        'data'][0] == x_dpt_from_table.__sycl_usm_array_interface__['data'][0]
    assert x_dpt.shape == x_dpt_from_table.shape
    assert x_dpt.strides == x_dpt_from_table.strides
    assert x_dpt.dtype == x_dpt_from_table.dtype
    assert x_dpt.flags.f_contiguous
    assert x_dpt_from_table.flags.f_contiguous
