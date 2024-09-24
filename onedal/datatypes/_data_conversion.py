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

import warnings

import numpy as np

from daal4py.sklearn._utils import make2d
from onedal import _backend, _is_dpc_backend

from .._config import _get_config, _set_config
from .._device_offload import dpctl_available, dpnp_available
from ..utils import _is_csr

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp


def _apply_and_pass(func, *args):
    if len(args) == 1:
        return func(args[0])
    return tuple(map(func, args))


def from_table(*args):
    return _apply_and_pass(convert_one_from_table, *args)


# TODO:
# add warnings if no dpc backend.
# TODO:
# sparse for sua data.
def convert_one_from_table(table):
    usm_iface = (
        getattr(table, "__sycl_usm_array_interface__", None) if _is_dpc_backend else None
    )
    if usm_iface and dpctl_available:
        result = dpt.asarray(table)
        if dpnp_available and _get_config()["_dpnp_output"]:
            result = dpnp.asarray(result)
        return result

    if not _is_csr(arg):
        arg = make2d(arg)
    return _backend.to_table(arg)


# TODO:
# add warnings if no dpc backend.
# TODO:
# sparse for sua data.
def convert_one_to_table(arg):
    usm_iface = (
        getattr(arg, "__sycl_usm_array_interface__", None) if _is_dpc_backend else None
    )
    if usm_iface:
        if dpnp_available and isinstance(arg, dpnp.ndarray):
            _set_config(_dpnp_output=True)
        return _backend.sua_iface_to_table(arg)

    if not _is_csr(arg):
        arg = make2d(arg)
    return _backend.to_table(arg)


def to_table(*args):
    return _apply_and_pass(convert_one_to_table, *args)


if _is_dpc_backend:
    from ..common._policy import _HostInteropPolicy

    def _convert_to_supported(policy, *data):
        def func(x):
            return x

        # CPUs support FP64 by default
        if isinstance(policy, _HostInteropPolicy):
            return _apply_and_pass(func, *data)

        # It can be either SPMD or DPCPP policy
        device = policy._queue.sycl_device

        def convert_or_pass(x):
            if (x is not None) and (x.dtype == np.float64):
                warnings.warn(
                    "Data will be converted into float32 from "
                    "float64 because device does not support it",
                    RuntimeWarning,
                )
                return x.astype(np.float32)
            else:
                return x

        if not device.has_aspect_fp64:
            func = convert_or_pass

        return _apply_and_pass(func, *data)

else:

    def _convert_to_supported(policy, *data):
        def func(x):
            return x

        return _apply_and_pass(func, *data)
