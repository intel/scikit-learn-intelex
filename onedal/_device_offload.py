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

from collections.abc import Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Optional

import numpy as np
from sklearn import get_config

from ._config import _get_config
from .utils._array_api import _asarray, _is_numpy_namespace
from .utils._dpep_helpers import dpctl_available, dpnp_available

if dpctl_available:
    from dpctl import SyclQueue
    from dpctl.memory import MemoryUSMDevice, as_usm_memory
    from dpctl.tensor import usm_ndarray
else:
    from onedal import _dpc_backend

    SyclQueue = getattr(_dpc_backend, "SyclQueue", None)


class SyclQueueManager:
    """Manage global and data SyclQueues"""

    # single instance of global queue
    __global_queue = None

    @staticmethod
    def __create_sycl_queue(target):
        if SyclQueue is None:
            # we don't have SyclQueue support
            return None
        if target is None:
            return None
        if isinstance(target, SyclQueue):
            return target
        if isinstance(target, (str, int)):
            return SyclQueue(target)
        raise ValueError(f"Invalid queue or device selector {target=}.")

    @staticmethod
    def get_global_queue():
        """Get the global queue. Retrieve it from the config if not set."""
        if (queue := SyclQueueManager.__global_queue) is not None:
            if not isinstance(queue, SyclQueue):
                raise ValueError("Global queue is not a SyclQueue object.")
            return queue

        target = _get_config()["target_offload"]
        if target == "auto":
            # queue will be created from the provided data to each function call
            return None

        q = SyclQueueManager.__create_sycl_queue(target)
        SyclQueueManager.update_global_queue(q)
        return q

    @staticmethod
    def remove_global_queue():
        """Remove the global queue."""
        SyclQueueManager.__global_queue = None

    @staticmethod
    def update_global_queue(queue):
        """Update the global queue."""
        queue = SyclQueueManager.__create_sycl_queue(queue)
        SyclQueueManager.__global_queue = queue

    @staticmethod
    def from_data(*data):
        """Extract the queue from provided data. This updates the global queue as well."""
        for item in data:
            # iterate through all data objects, extract the queue, and verify that all data objects are on the same device

            # get the `usm_interface` - the C++ implementation might throw an exception if the data type is not supported
            try:
                usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
            except RuntimeError as e:
                if "SUA interface" in str(e):
                    # ignore SUA interface errors and move on
                    continue
                else:
                    # unexpected, re-raise
                    raise e

            if usm_iface is None:
                # no interface found - try next data object
                continue

            # extract the queue
            global_queue = SyclQueueManager.get_global_queue()
            data_queue = usm_iface["syclobj"]
            if not data_queue:
                # no queue, i.e. host data, no more work to do
                continue

            # update the global queue if not set
            if global_queue is None:
                SyclQueueManager.update_global_queue(data_queue)
                global_queue = data_queue

            # if either queue points to a device, assert it's always the same device
            data_dev = data_queue.sycl_device
            global_dev = global_queue.sycl_device
            if (data_dev and global_dev) is not None and data_dev != global_dev:
                raise ValueError(
                    "Data objects are located on different target devices or not on selected device."
                )

        # after we went through the data, global queue is updated and verified (if any queue found)
        return SyclQueueManager.get_global_queue()

    @staticmethod
    @contextmanager
    def manage_global_queue(queue, *args):
        """
        Context manager to manage the global SyclQueue.

        This context manager updates the global queue with the provided queue,
        verifies that all data objects are on the same device, and restores the
        original queue after work is done.
        Note: For most applications, the original queue should be `None`, but
              if there are nested calls to `manage_global_queue()`, it is
              important to restore the outer queue, rather than setting it to
              `None`.

        Parameters:
        queue (SyclQueue or None): The queue to set as the global queue. If None,
                                   the global queue will be determined from the provided data.
        *args: Additional data objects to verify their device placement.

        Yields:
        SyclQueue: The global queue after verification.
        """
        original_queue = SyclQueueManager.get_global_queue()
        try:
            # update the global queue with what is provided, it can be None, then we will get it from provided data
            SyclQueueManager.update_global_queue(queue)
            # find the queues in data using SyclQueueManager to verify that all data objects are on the same device
            yield SyclQueueManager.from_data(*args)
        finally:
            # restore the original queue
            SyclQueueManager.update_global_queue(original_queue)


def supports_queue(func):
    """
    Decorator that updates the global queue based on provided queue and global configuration.
    If a `queue` keyword argument is provided in the decorated function, its value will be used globally.
    If no queue is provided, the global queue will be updated from the provided data.
    In either case, all data objects are verified to be on the same device (or on host).
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        queue = kwargs.get("queue", None)
        with SyclQueueManager.manage_global_queue(queue, *args) as queue:
            kwargs["queue"] = queue
            result = func(self, *args, **kwargs)
        return result

    return wrapper


if dpnp_available:
    import dpnp

    from .utils._array_api import _convert_to_dpnp


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


def _transfer_to_host(*data):
    has_usm_data, has_host_data = False, False

    host_data = []
    for item in data:
        usm_iface = getattr(item, "__sycl_usm_array_interface__", None)
        array_api = getattr(item, "__array_namespace__", lambda: None)()
        if usm_iface is not None:
            if not dpctl_available:
                raise RuntimeError(
                    "dpctl need to be installed to work "
                    "with __sycl_usm_array_interface__"
                )

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
    return has_usm_data, host_data


def _get_host_inputs(*args, **kwargs):
    _, hostargs = _transfer_to_host(*args)
    _, hostvalues = _transfer_to_host(*kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return hostargs, hostkwargs


def _run_on_device(func, obj=None, *args, **kwargs):
    if obj is not None:
        return func(obj, *args, **kwargs)
    return func(*args, **kwargs)


def support_input_format(freefunc=False, queue_param=True):
    """
    Converts and moves the output arrays of the decorated function
    to match the input array type and device.
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
            if len(args) == 0 and len(kwargs) == 0:
                return _run_on_device(func, obj, *args, **kwargs)

            hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            if hostkwargs.get("queue") is None:
                # no queue provided, get it from the data
                data_queue = SyclQueueManager.from_data(*hostargs)
                if queue_param:
                    # if queue_param requested, add it to the hostkwargs
                    hostkwargs["queue"] = data_queue
            else:
                # use the provided queue
                data_queue = hostkwargs["queue"]

            data = (*args, *kwargs.values())
            result = _run_on_device(func, obj, *hostargs, **hostkwargs)

            if data_queue is not None:
                result = _copy_to_usm(data_queue, result)
                if dpnp_available and isinstance(data[0], dpnp.ndarray):
                    result = _convert_to_dpnp(result)
                return result

            if not get_config().get("transform_output"):
                input_array_api = getattr(data[0], "__array_namespace__", lambda: None)()
                if input_array_api:
                    input_array_api_device = data[0].device
                    result = _asarray(
                        result, input_array_api, device=input_array_api_device
                    )
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
