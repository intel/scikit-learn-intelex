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


# TODO:
# add warnings if no dpc backend.
# TODO:
# sparse for sua data.
# TODO:
# update it for each of the datafrmae format.
# TODO:
# update func use with args and kwargs with _apply_and_pass.
def convert_one_from_table(table, sycl_queue=None, sua_iface=None, xp=None):
    # Currently only `__sycl_usm_array_interface__` protocol used to
    # convert into dpnp/dpctl tensors.
    if sua_iface:
        if (
            sycl_queue
            and sycl_queue.sycl_device.is_cpu
            and table.__sycl_usm_array_interface__["syclobj"] is None
        ):
            # OneDAL returns tables with None sycl queue for CPU sycl queue inputs.
            # This workaround is necessary for the functional preservation
            # of the compute-follows-data execution.
            # Host tables first converted into numpy.narrays and then to array from xp
            # namespace.
            return xp.asarray(
                _backend.from_table(table), usm_type="device", sycl_queue=sycl_queue
            )
        else:
            xp_name = xp.__name__
            if dpnp_available and xp_name == "dpnp":
                # By default DPNP ndarray created with a copy.
                # TODO:
                # investigate why dpnp.array(table, copy=False) doesn't work.
                # Work around with using dpctl.tensor.asarray.
                return xp.array(dpt.asarray(table), copy=False)
            else:
                return xp.asarray(table)

    return _backend.from_table(table)


# TODO:
# add warnings if no dpc backend.
# TODO:
# sparse for sua data.
def convert_one_to_table(arg, sua_iface=None):
    if sua_iface and _is_dpc_backend:
        return _backend.sua_iface_to_table(arg)

    if not _is_csr(arg):
        arg = make2d(arg)
    return _backend.to_table(arg)


def from_table(*args):
    return _apply_and_pass(convert_one_from_table, *args)


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
