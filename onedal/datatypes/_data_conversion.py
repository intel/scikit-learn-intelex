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

import numpy as np
import warnings

from onedal import _is_dpc_backend
from onedal import _backend
from daal4py.sklearn._utils import make2d

try:
    import dpctl
    import dpctl.tensor as dpt
    dpctl_available = dpctl.__version__ >= '0.14'
except ImportError:
    dpctl_available = False


def _apply_and_pass(func, *args):
    if len(args) == 1:
        return func(args[0])
    return tuple(map(func, args))


def from_table(*args):
    return _apply_and_pass(_backend.from_table, *args)


def convert_one_to_table(arg):
    if dpctl_available:
        if isinstance(arg, dpt.usm_ndarray):
            return _backend.dpctl_to_table(arg)
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
                warnings.warn("Data will be converted into float32 from "
                              "float64 because device does not support it",
                              RuntimeWarning, )
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


#TODO: Update with dpctl
#For now it will fall back to numpy
def _convert_one_to_dataframe(policy, x):
    is_numpy = isinstance(x, np.ndarray)
    if (x is None) or is_numpy:
        return x
    else:
        return np.asarray(x)


def _convert_to_dataframe(policy, *data):
    def _convert_one(x):
        return _convert_one_to_dataframe(policy, x)
    return _apply_and_pass(_convert_one, *data)
