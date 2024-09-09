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

from daal4py.sklearn._utils import get_dtype
from daal4py.sklearn._utils import make2d as d4p_make2d
from onedal import _backend, _is_dpc_backend

from ..utils import _is_csr

# from ..utils._array_api import get_namespace
from ..utils._array_api import _is_numpy_namespace


# TODO:
# move to proper module.
# TODO
# def make2d(arg, xp=None, is_array_api_compliant=None):
def make2d(arg, xp=None):
    if xp and not _is_numpy_namespace(xp) and arg.ndim == 1:
        return xp.reshape(arg, (arg.size, 1)) if arg.ndim == 1 else arg
    # TODO:
    # reimpl via is_array_api_compliant usage.
    return d4p_make2d(arg)


# TODO:
# remove such kind of func calls
def _apply_and_pass(func, *args, **kwargs):
    # TODO:
    # refactor.
    if len(args) == 1:
        return func(args[0])
    return tuple(map(func, args, kwargs))


def convert_one_from_table(arg, xp=None, queue=None, array_api_device=None):
    # TODO:
    # use `array_api_device`.
    result = _backend.from_table(arg)
    if xp:
        if xp.__name__ in {"dpctl", "dpctl.tensor"}:
            result = xp.asarray(arg, sycl_queue=queue) if queue else xp.asarray(arg)
        elif not _is_numpy_namespace(xp):
            results = xp.asarray(result)
    return result


def convert_one_to_table(arg, xp=None):
    if not _is_csr(arg):
        if xp and not _is_numpy_namespace(xp):
            arg = np.asarray(arg)
        # TODO:
        # Check. Probably should be removed from here
        # Not really realted with converting to table.
        arg = make2d(arg)
    return _backend.to_table(arg)


def from_table(*args, xp=None, queue=None, array_api_device=None):
    return _apply_and_pass(convert_one_from_table, *args)


def to_table(*args, xp=None):
    return _apply_and_pass(convert_one_to_table, *args, xp=xp)


if _is_dpc_backend:
    from ..common._policy import _HostInteropPolicy

    def _convert_to_supported(policy, *data, xp=None):
        if xp is None:
            xp = np

        def func(x):
            return x

        # CPUs support FP64 by default
        if isinstance(policy, _HostInteropPolicy):
            return _apply_and_pass(func, *data)

        # It can be either SPMD or DPCPP policy
        device = policy._queue.sycl_device

        def convert_or_pass(x):
            if (x is not None) and (x.dtype == xp.float64):
                warnings.warn(
                    "Data will be converted into float32 from "
                    "float64 because device does not support it",
                    RuntimeWarning,
                )
                return x.astype(xp.float32)
            else:
                return x

        if not device.has_aspect_fp64:
            func = convert_or_pass

        return _apply_and_pass(func, *data)

else:

    def _convert_to_supported(policy, *data, xp=None):
        def func(x):
            return x

        return _apply_and_pass(func, *data)
