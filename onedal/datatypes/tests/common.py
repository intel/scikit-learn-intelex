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

from onedal._device_offload import dpctl_available, dpnp_available

if dpnp_available:
    import dpnp

if dpctl_available:
    from dpctl.tensor import usm_ndarray


def _assert_tensor_attr(actual, desired, order):
    is_usm_tensor = lambda x: isinstance(x, dpnp.ndarray) or isinstance(x, usm_ndarray)
    assert is_usm_tensor(actual)
    assert is_usm_tensor(desired)
    # dpctl.tensor is the dpnp.ndarrays's core tensor structure along
    # with advanced device management. Convert dpnp to dpctl.tensor with zero copy.
    get_tensor = lambda x: (
        x.get_array() if dpnp_available and isinstance(x, dpnp.ndarray) else x
    )
    # Now DPCtl tensors
    actual = get_tensor(actual)
    desired = get_tensor(desired)

    assert actual.shape == desired.shape
    assert actual.strides == desired.strides
    assert actual.dtype == desired.dtype
    if order == "F":
        assert actual.flags.f_contiguous
        assert desired.flags.f_contiguous
        assert actual.flags.f_contiguous == desired.flags.f_contiguous
    else:
        assert actual.flags.c_contiguous
        assert desired.flags.c_contiguous
        assert actual.flags.c_contiguous == desired.flags.c_contiguous
    assert actual.flags == desired.flags
    assert actual.sycl_queue == desired.sycl_queue
    # TODO:
    # check better way to check usm ptrs.
    assert actual.usm_data._pointer == desired.usm_data._pointer


# TODO:
# remove skip_syclobj and skip_data_1 params.
# def _assert_sua_iface_fields(actual, desired):
def _assert_sua_iface_fields(actual, desired, skip_syclobj=False, skip_data_1=False):
    assert hasattr(actual, "__sycl_usm_array_interface__")
    assert hasattr(desired, "__sycl_usm_array_interface__")
    actual_sua_iface = actual.__sycl_usm_array_interface__
    desired_sua_iface = desired.__sycl_usm_array_interface__
    # TODO:
    # do value checks by the dict keys in for.
    assert actual_sua_iface["data"][0] == desired_sua_iface["data"][0]
    # TODO:
    # remove this condition/param.
    if not skip_data_1:
        assert actual_sua_iface["data"][1] == desired_sua_iface["data"][1]
    assert actual_sua_iface["shape"] == desired_sua_iface["shape"]
    if not actual_sua_iface["strides"] and not desired_sua_iface["strides"]:
        # None to indicate a C-style contiguous 1D array.
        # onedal4py constructs __sycl_usm_array_interface__["strides"] with
        # real values.
        assert actual_sua_iface["strides"] == desired_sua_iface["strides"]
    assert actual_sua_iface["version"] == desired_sua_iface["version"]
    assert actual_sua_iface["typestr"] == desired_sua_iface["typestr"]
    if not skip_syclobj:
        # TODO:
        # comment and the conditions to check values.
        assert actual_sua_iface["syclobj"]._get_capsule() == desired_sua_iface["syclobj"]
