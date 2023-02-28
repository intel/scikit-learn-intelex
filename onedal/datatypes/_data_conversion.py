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

from onedal import _backend
from daal4py.sklearn._utils import make2d


def from_table(*args):
    if len(args) == 1:
        return _backend.from_table(args[0])
    return (_backend.from_table(item) for item in args)


def convert_one_to_table(arg):
    arg = make2d(arg)
    return _backend.to_table(arg)


def to_table(*args):
    if len(args) == 1:
        return convert_one_to_table(args[0])
    return (convert_one_to_table(item) for item in args)


from onedal import _is_dpc_backend

if _is_dpc_backend:
    import numpy as np

    from ..common._policy import _HostInteropPolicy

    def _convert_to_supported_impl(policy, *data):
        # CPUs support FP64 by default
        if isinstance(policy, _HostInteropPolicy):
            return data

        # It can be either SPMD or DPCPP policy
        device = policy._queue.sycl_device

        def convert_or_pass(x):
            if x.dtype is np.float64:
                return x.astype(np.float32)
            else:
                return x

        if not device.has_aspect_fp64:
            return (convert_or_pass(x) for x in data)
        else:
            return data
else:
    def _convert_to_supported_impl(policy, *data):
        return data


def _convert_to_supported(policy, *data):
    res = _convert_to_supported_impl(policy, *data)

    if len(data) == 1:
        return res[0]
    else:
        return res
