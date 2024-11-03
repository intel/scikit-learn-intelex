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
import scipy.sparse as sp

from onedal import _backend, _is_dpc_backend


def make2d(X):
    # generalized for array-like inputs
    # dpnp -1 indexing is broken, use size
    if hasattr(X, "reshape") and hasattr(X, "ndim") and X.ndim == 1:
        return X.reshape((X.size, 1))
    if np.isscalar(X):
        return np.atleast_2d(X)
    return X


def _apply_and_pass(func, *args, **kwargs):
    if len(args) == 1:
        return func(args[0], **kwargs)
    return tuple(map(lambda arg: func(arg, **kwargs), args))


def convert_one_to_table(arg):
    return _backend.to_table(arg if sp.issparse(arg) else make2d(arg))


def to_table(*args):
    return _apply_and_pass(convert_one_to_table, *args)


if _is_dpc_backend:

    try:
        import dpnp

        def _onedal_gpu_table_to_array(table, xp=None):
            # By default DPNP ndarray created with a copy.
            # TODO:
            # investigate why dpnp.array(table, copy=False) doesn't work.
            # Work around with using dpctl.tensor.asarray.
            if xp == dpnp:
                return dpnp.array(dpnp.dpctl.tensor.asarray(table), copy=False)
            else:
                return xp.asarray(table)

    except ImportError:

        def _onedal_gpu_table_to_array(table, xp=None):
            return xp.asarray(table)

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

    def convert_one_from_table(table, sycl_queue=None, sua_iface=None, xp=None):
        # Currently only `__sycl_usm_array_interface__` protocol used to
        # convert into dpnp/dpctl tensors.
        if sua_iface:
            if (
                sycl_queue
                and sycl_queue.sycl_device.is_cpu
                and table.__sycl_usm_array_interface__["syclobj"] is None
            ):
                # oneDAL returns tables with None sycl queue for CPU sycl queue inputs.
                # This workaround is necessary for the functional preservation
                # of the compute-follows-data execution.
                # Host tables first converted into numpy.narrays and then to array from xp
                # namespace.
                return xp.asarray(
                    _backend.from_table(table), usm_type="device", sycl_queue=sycl_queue
                )
            else:
                return _onedal_gpu_table_to_array(table, xp=xp)

        return _backend.from_table(table)

else:

    def _convert_to_supported(policy, *data):
        def func(x):
            return x

        return _apply_and_pass(func, *data)

    def convert_one_from_table(table, sycl_queue=None, sua_iface=None, xp=None):
        # Currently only `__sycl_usm_array_interface__` protocol used to
        # convert into dpnp/dpctl tensors.
        if sua_iface:
            raise RuntimeError(
                "SYCL usm array conversion from table requires the DPC backend"
            )
        return _backend.from_table(table)


def from_table(*args, sycl_queue=None, sua_iface=None, xp=None):
    return _apply_and_pass(
        convert_one_from_table, *args, sycl_queue=sycl_queue, sua_iface=sua_iface, xp=xp
    )
