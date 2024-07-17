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

from functools import wraps

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False

try:
    from sklearnex._device_offload import (
        _copy_to_usm,
        _get_global_queue,
        _transfer_to_host,
    )

    _sklearnex_available = True
except ImportError:
    import logging

    logging.warning("Device support requires " "Intel(R) Extension for Scikit-learn*.")
    _sklearnex_available = False


def _get_host_inputs(*args, **kwargs):
    q = _get_global_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return q, hostargs, hostkwargs


def _extract_usm_iface(*args, **kwargs):
    allargs = (*args, *kwargs.values())
    if len(allargs) == 0:
        return None
    return getattr(allargs[0], "__sycl_usm_array_interface__", None)


def _run_on_device(func, obj=None, *args, **kwargs):
    if obj is not None:
        return func(obj, *args, **kwargs)
    return func(*args, **kwargs)


<<<<<<< HEAD
def support_usm_ndarray(freefunc=False):
    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            if _sklearnex_available:
                usm_iface = _extract_usm_iface(*args, **kwargs)
                data_queue, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
=======
if dpnp_available:

    def _convert_to_dpnp(array):
        if isinstance(array, usm_ndarray):
            return dpnp.array(array, copy=False)
        elif isinstance(array, Iterable):
            for i in range(len(array)):
                array[i] = _convert_to_dpnp(array[i])
        return array


def support_usm_ndarray(freefunc=False, queue_param=True):
    """
    Handles USMArray input. Puts SYCLQueue from data to decorated function arguments.
    Converts output of decorated function to dpctl.tensor/dpnp.ndarray if input was of this type.

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
            usm_iface = _extract_usm_iface(*args, **kwargs)
            data_queue, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            if queue_param and not (
                "queue" in hostkwargs and hostkwargs["queue"] is not None
            ):
>>>>>>> 69d147fc (FIX: prevent `support_usm_ndarray` from changing queue if explicitly provided. (#1940))
                hostkwargs["queue"] = data_queue
                result = _run_on_device(func, obj, *hostargs, **hostkwargs)
                if usm_iface is not None and hasattr(result, "__array_interface__"):
                    result = _copy_to_usm(data_queue, result)
                    if (
                        dpnp_available
                        and len(args) > 0
                        and isinstance(args[0], dpnp.ndarray)
                    ):
                        result = dpnp.array(result, copy=False)
                return result
            return _run_on_device(func, obj, *args, **kwargs)

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
