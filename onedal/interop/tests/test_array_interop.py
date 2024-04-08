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

from onedal.interop import from_array, is_array_entity, to_array
from onedal.tests.utils._device_selection import get_queues

from .wrappers import get_dtype_list, wrap_entity

try:
    import dpctl
    import dpctl.tensor as dpt

    dpctl_available = dpctl.__version__ >= "0.14"
except ImportError:
    dpctl_available = False


def check_strides(value, truth):
    value = (1,) if value is None else value
    truth = (1,) if truth is None else truth
    return value == truth


@pytest.mark.skipif(not dpctl_available, reason="requires dpctl>=0.14")
@pytest.mark.parametrize("queue", get_queues("cpu,gpu"))
@pytest.mark.parametrize("backend", ["sua", "dlpack", "native"])
@pytest.mark.parametrize("count", [1, 5, 10, 50, 1000, 4999, 100001])
@pytest.mark.parametrize("dtype", get_dtype_list())
def test_device_array_functionality(queue, backend, count, dtype):
    generator = np.random.Generator(np.random.MT19937(count**2))
    numpy_array = generator.integers(0, 777, count).astype(dtype=dtype)
    dpctl_tensor = dpt.asarray(numpy_array, usm_type="device", sycl_queue=queue)
    dpctl_sua = dpctl_tensor.__sycl_usm_array_interface__
    tensor_device = dpctl_tensor.__dlpack_device__()

    wrapped_tensor = wrap_entity(dpctl_tensor, backend)
    onedal_array = to_array(wrapped_tensor)

    assert is_array_entity(onedal_array)
    del dpctl_tensor, wrapped_tensor

    assert onedal_array.get_data() == dpctl_sua["data"][0]
    assert onedal_array.get_count() == dpctl_sua["shape"][0]

    return_array = from_array(onedal_array)

    np.testing.assert_equal(numpy_array, return_array)


def check_by_sampling(generator, numpy_array, onedal_array):
    count = len(onedal_array)
    assert count == len(numpy_array)
    sample_count = int(max(count / 10, min(count, 10)))
    sample_indices = generator.integers(0, count, sample_count)
    onedal_samples = [int(onedal_array[s]) for s in sample_indices]
    np.testing.assert_equal(onedal_samples, numpy_array[sample_indices])


# TODO:
# update test.
@pytest.mark.parametrize("backend", ["dlpack", "native", "buffer"])
@pytest.mark.parametrize("count", [1, 5, 10, 50, 1000, 4999, 100001])
@pytest.mark.parametrize("dtype", get_dtype_list())
def test_host_array_functionality(backend, count, dtype):
    generator = np.random.Generator(np.random.MT19937(count))
    numpy_array = generator.integers(0, 888, count).astype(dtype=dtype)
    numpy_device = numpy_array.__dlpack_device__()
    numpy_iface = numpy_array.__array_interface__

    wrapped_tensor = wrap_entity(numpy_array, backend)

    # Writing something to array results
    # in data copy. Wrapper specific
    # and doesn't affect user
    if backend == "buffer":
        ptr = wrapped_tensor.buffer_info()[0]
    else:
        ptr = numpy_iface["data"][0]

    onedal_array = to_array(wrapped_tensor)
    assert is_array_entity(onedal_array)
    del wrapped_tensor

    assert onedal_array.get_data() == ptr
    assert onedal_array.get_count() == numpy_iface["shape"][0]

    check_by_sampling(generator, numpy_array, onedal_array)

    return_array = from_array(onedal_array)
    return_iface = return_array.__array_interface__

    assert return_iface["data"][0] == ptr
    assert return_iface["shape"] == numpy_iface["shape"]
    assert return_array.__dlpack_device__() == numpy_device
    assert return_iface["typestr"] == numpy_iface["typestr"]
    assert check_strides(return_iface["strides"], numpy_iface["strides"])
