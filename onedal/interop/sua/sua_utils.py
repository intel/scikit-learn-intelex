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

import logging

from onedal.interop.utils import check_attr

convert_sua_impl = None

try:
    import dpctl.tensor as dpt

    # TODO: Make sure that it will return
    # SYCL-native entity in the future
    def __convert_sua_impl(sua):
        result = dpt.asarray(sua)
        return dpt.asnumpy(result)

    if convert_sua_impl is None:
        convert_sua_impl = __convert_sua_impl
except ImportError:
    logging.info("Unable to load DPCTL")

try:
    import dpnp

    # TODO: Make sure that it will return
    # SYCL-native entity in the future
    def __convert_sua_impl(sua):
        result = dpnp.asarray(sua)
        return dpnp.asnumpy(result)

    if convert_sua_impl is None:
        convert_sua_impl = __convert_sua_impl

except ImportError:
    logging.info("Unable to load DPNP")


# TODO: Check supported versions of iface
def is_valid_sua_iface(iface: dict) -> bool:
    if isinstance(iface, dict):
        check_version = lambda v: v == 1
        return check_attr(iface, "version", check_version)
    return False


def is_sua_entity(entity) -> bool:
    iface_attr: str = "__sycl_usm_array_interface__"
    if hasattr(entity, iface_attr):
        iface: dict = getattr(entity, iface_attr)
        return is_valid_sua_iface(iface)
    return False


def get_sua_iface(entity) -> dict:
    assert is_sua_entity(entity)
    return entity.__sycl_usm_array_interface__


# TODO: Make sure that it will return
# SYCL-native entity in the future
def convert_sua(sua):
    assert is_sua_entity(sua)
    if convert_sua_impl is not None:
        return convert_sua_impl(sua)
    return None


def is_nd(entity, n: int = 1) -> bool:
    if is_sua_entity(entity):
        iface = get_sua_iface(entity)
        shape = iface["shape"]
        return len(shape) == n
    return False
