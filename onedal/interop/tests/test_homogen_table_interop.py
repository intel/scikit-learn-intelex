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

import onedal
from onedal.interop.homogen_table import (
    from_homogen_table,
    is_homogen_entity,
    to_homogen_table,
)
from onedal.interop.sua import is_sua_entity
from onedal.tests.utils._device_selection import get_queues

from .wrappers import get_dtype_list, wrap_entity

try:
    import dpctl
    import dpctl.tensor as dpt

    dpctl_available = dpctl.__version__ >= "0.14"
except ImportError:
    dpctl_available = False

table_dimensions = [
    (1, 1),
    (1, 10),
    (11, 1),
    (3, 9),
    (21, 17),
    (1001, 1),
    (999, 1),
    (888, 777),
    (123, 999),
]

data_layout = onedal._backend.data_management.data_layout


def check_table_dimensions(table, shape, transpose):
    assert table.get_row_count() == shape[0]
    assert table.get_column_count() == shape[1]

    is_simple = shape[0] == 1
    is_simple = is_simple or shape[1] == 1

    curr_layout = table.get_data_layout()
    if transpose and not is_simple:
        column_major = data_layout.column_major
        assert curr_layout == column_major
    else:
        row_major = data_layout.row_major
        assert curr_layout == row_major


@pytest.mark.skipif(not dpctl_available, reason="requires dpctl>=0.14")
@pytest.mark.parametrize("queue", get_queues("cpu,gpu"))
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("backend", ["sua", "dlpack", "native"])
@pytest.mark.parametrize("shape", table_dimensions)
@pytest.mark.parametrize("dtype", get_dtype_list())
def test_device_array_functionality(queue, backend, transpose, shape, dtype):
    generator = np.random.Generator(np.random.MT19937(min(shape)))
    numpy_array = generator.integers(0, 555, shape).astype(dtype=dtype)
    numpy_array = numpy_array.T if transpose else numpy_array
    dpctl_tensor = dpt.asarray(numpy_array, usm_type="device", sycl_queue=queue)
    dpctl_sua = dpctl_tensor.__sycl_usm_array_interface__
    tensor_device = dpctl_tensor.__dlpack_device__()

    wrapped_tensor = wrap_entity(dpctl_tensor, backend)
    onedal_table = to_homogen_table(wrapped_tensor)

    assert is_homogen_entity(onedal_table)
    del dpctl_tensor, wrapped_tensor

    curr_shape = numpy_array.shape
    check_table_dimensions(onedal_table, curr_shape, transpose)

    return_table = from_homogen_table(onedal_table)

    if is_sua_entity(return_table):
        return_sua = return_table.__sycl_usm_array_interface__

        assert return_sua["shape"] == dpctl_sua["shape"]
        assert return_sua["typestr"] == dpctl_sua["typestr"]
        assert return_sua["data"][0] == dpctl_sua["data"][0]
        assert return_table.__dlpack_device__() == tensor_device
    else:
        np.testing.assert_equal(numpy_array, return_table)


@pytest.mark.parametrize("backend", ["dlpack", "native"])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("shape", table_dimensions)
@pytest.mark.parametrize("dtype", get_dtype_list())
def test_host_homogen_table_functionality(backend, transpose, shape, dtype):
    generator = np.random.Generator(np.random.MT19937(sum(shape)))
    numpy_array = generator.integers(0, 999, shape).astype(dtype=dtype)
    numpy_array = numpy_array.T if transpose else numpy_array

    numpy_device = numpy_array.__dlpack_device__()
    numpy_iface = numpy_array.__array_interface__

    wrapped_tensor = wrap_entity(numpy_array, backend)
    onedal_table = to_homogen_table(wrapped_tensor)
    assert is_homogen_entity(onedal_table)

    curr_shape = numpy_array.shape
    check_table_dimensions(onedal_table, curr_shape, transpose)

    return_table = from_homogen_table(onedal_table)
    return_iface = return_table.__array_interface__

    assert return_iface["shape"] == numpy_iface["shape"]
    assert return_table.__dlpack_device__() == numpy_device
    assert return_iface["typestr"] == numpy_iface["typestr"]
    assert return_iface["data"][0] == numpy_iface["data"][0]
