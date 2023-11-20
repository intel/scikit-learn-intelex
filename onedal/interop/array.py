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

import onedal
import onedal.interop.buffer as buffer
import onedal.interop.dlpack as dlpack
import onedal.interop.sua as sua
from onedal.common._policy import _are_equal_devices, _HostInteropPolicy
from onedal.interop.utils import is_host_policy

make_array = onedal._backend.data_management.make_array


def is_python_array(entity) -> bool:
    conditions = [sua.is_sua_array, dlpack.is_dlpack_array, buffer.is_buffer_array]
    return any(map(lambda check: check(entity), conditions))


def is_native_array(entity) -> bool:
    attr: str = "__is_onedal_array__"
    if hasattr(entity, attr):
        return getattr(entity, attr)
    return False


def is_array_entity(entity) -> bool:
    conditions = [is_python_array, is_native_array]
    return any(map(lambda check: check(entity), conditions))


def to_array_python(entity):
    assert is_python_array(entity)
    if sua.is_sua_array(entity):
        result = sua.to_array(entity)
    elif dlpack.is_dlpack_array(entity):
        result = dlpack.to_array(entity)
    elif buffer.is_buffer_array(entity):
        result = buffer.to_array(entity)
    else:
        raise ValueError("Unable to convert to array from python")
    assert is_native_array(result)
    return result


def to_array_native(entity):
    assert is_native_array(entity)
    return make_array(entity)


def to_array(entity):
    assert is_array_entity(entity)
    if is_native_array(entity):
        result = to_array_native(entity)
    elif is_python_array(entity):
        result = to_array_python(entity)
    else:
        raise ValueError("Not able to convert to array")
    assert is_native_array(result)
    return result


def to_typed_array(x, dtypes=[np.float32]):
    result = x
    if x.dtype not in list(dtypes):
        result = x.astype(dtypes[0])
    assert result.dtype in dtypes
    return to_array(result)


# TODO: Implement smarter logic
# for better performance
def to_common_policy(*xs) -> tuple:
    if _are_equal_devices(xs):
        return xs
    else:
        policy = _HostInteropPolicy()
        return (x.to_policy(policy) for x in xs)


def from_array_native(array):
    assert is_native_array(array)
    policy = array.get_policy()
    device = policy.get_device_name()
    if is_host_policy(policy):
        result = buffer.from_array(array)
    elif device in ["cpu", "gpu"]:
        result = sua.from_array(array)
    else:
        raise ValueError("Unable to convert from array to python")
    assert is_python_array(result)
    return result


def from_array_python(array):
    assert is_python_array(array)
    return array


def from_array(array):
    assert is_array_entity(array)
    if is_native_array(array):
        result = from_array_native(array)
    elif is_python_array(array):
        result = from_array_python(array)
    else:
        raise ValueError("Unable to convert from array")
    assert is_python_array(result)
    return result
