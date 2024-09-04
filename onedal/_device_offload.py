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
from collections.abc import Iterable
from functools import wraps

import numpy as np

from ._config import _get_config
from .utils._array_api import _asarray, _is_numpy_namespace

try:
    from dpctl import SyclQueue
    from dpctl.memory import MemoryUSMDevice, as_usm_memory
    from dpctl.tensor import usm_ndarray

    dpctl_available = True
except ImportError:
    dpctl_available = False

try:
    import dpnp

    from .utils._array_api import _convert_to_dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False


class DummySyclQueue:
    """This class is designed to act like dpctl.SyclQueue
    to allow device dispatching in scenarios when dpctl is not available"""

    class DummySyclDevice:
        def __init__(self, filter_string):
            self._filter_string = filter_string
            self.is_cpu = "cpu" in filter_string
            self.is_gpu = "gpu" in filter_string
            self.has_aspect_fp64 = self.is_cpu

            if not (self.is_cpu):
                logging.warning(
                    "Device support is limited. "
                    "Please install dpctl for full experience"
                )

        def get_filter_string(self):
            return self._filter_string

    def __init__(self, filter_string):
        self.sycl_device = self.DummySyclDevice(filter_string)


def _copy_to_usm(queue, array):
    if not dpctl_available:
        raise RuntimeError(
            "dpctl need to be installed to work " "with __sycl_usm_array_interface__"
        )

    if hasattr(array, "__array__"):

        try:
            mem = MemoryUSMDevice(array.nbytes, queue=queue)
            mem.copy_from_host(array.tobytes())
            return usm_ndarray(array.shape, array.dtype, buffer=mem)
        except ValueError as e:
            # ValueError will raise if device does not support the dtype
            # retry with float32 (needed for fp16 and fp64 support issues)
            # try again as float32, if it is a float32 just raise the error.
            if array.dtype == np.float32:
                raise e
            return _copy_to_usm(queue, array.astype(np.float32))
    else:
        if isinstance(array, Iterable):
            array = [_copy_to_usm(queue, i) for i in array]
        return array


def _transfer_to_host(queue, *data):
    has_usm_data, has_host_data = False, False

    host_data = []
    for item in data:
        usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
        array_api = getattr(item, "__array_namespace__", print)()
        if usm_iface is not None:
            if not dpctl_available:
                raise RuntimeError(
                    "dpctl need to be installed to work "
                    "with __sycl_usm_array_interface__"
                )
            if queue is not None:
                if queue.sycl_device != usm_iface["syclobj"].sycl_device:
                    raise RuntimeError(
                        "Input data shall be located " "on single target device"
                    )
            else:
                queue = usm_iface["syclobj"]

            buffer = as_usm_memory(item).copy_to_host()
            order = "C"
            if usm_iface["strides"] is not None:
                if usm_iface["strides"][0] < usm_iface["strides"][1]:
                    order = "F"
            item = np.ndarray(
                shape=usm_iface["shape"],
                dtype=usm_iface["typestr"],
                buffer=buffer,
                order=order,
            )
            has_usm_data = True
        elif array_api and not _is_numpy_namespace(array_api):
            # `copy`` param for the `asarray`` is not setted.
            # The object is copied only if needed.
            item = np.asarray(item)
            has_host_data = True
        else:
            has_host_data = True

        mismatch_host_item = usm_iface is None and item is not None and has_usm_data
        mismatch_usm_item = usm_iface is not None and has_host_data

        if mismatch_host_item or mismatch_usm_item:
            raise RuntimeError("Input data shall be located on single target device")

        host_data.append(item)
    return queue, host_data


def _get_global_queue():
    target = _get_config()["target_offload"]

    QueueClass = DummySyclQueue if not dpctl_available else SyclQueue

    if target != "auto":
        if isinstance(target, QueueClass):
            return target
        return QueueClass(target)
    return None


def _get_host_inputs(*args, **kwargs):
    q = _get_global_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return q, hostargs, hostkwargs


def _run_on_device(func, obj=None, *args, **kwargs):
    if obj is not None:
        return func(obj, *args, **kwargs)
    return func(*args, **kwargs)


def support_array_api(freefunc=False, queue_param=True):
    """
    Handles Array API input. Converts output of decorated function
    to input Array API format on the same device.
    Puts SYCLQueue from data to decorated function arguments.

    Parameters
    ----------
    freefunc (bool) : Set to True if decorates free function.
    queue_param (bool) : Set to False if the decorated function has no `queue` parameter

    Notes
    -----
    Queue will not be changed if provided explicitly.
    """

    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            data = (*args, *kwargs.values())
            if len(data) == 0:
                return _run_on_device(func, obj, *args, **kwargs)
            data_queue, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            if queue_param and not (
                "queue" in hostkwargs and hostkwargs["queue"] is not None
            ):
                hostkwargs["queue"] = data_queue
            result = _run_on_device(func, obj, *hostargs, **hostkwargs)
            usm_iface = getattr(data[0], "__sycl_usm_array_interface__", None)
            if usm_iface is not None:
                result = _copy_to_usm(data_queue, result)
                if dpnp_available and isinstance(args[0], dpnp.ndarray):
                    result = _convert_to_dpnp(result)
                return result
            input_array_api = getattr(data[0], "__array_namespace__", print)()
            input_array_api_device = data[0].device if input_array_api else None
            if input_array_api:
                result = _asarray(result, input_array_api, device=input_array_api_device)
            return result

        if freefunc:

            @wraps(func)
            def wrapper_free(*args, **kwargs):
                return wrapper_impl(None, *args, **kwargs)

            return wrapper_free

        @wraps(func)
        def wrapper_with_self(self, *args, **kwargs):
            return wrapper_impl(self, *args, **kwargs)

        return wrapper_with_self

    return decorator
