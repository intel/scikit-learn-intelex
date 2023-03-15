#===============================================================================
# Copyright 2014 Intel Corporation
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
#===============================================================================

from functools import wraps

# TODO:
# will be replaced with common check.
try:
    from dpctl import SyclQueue
    from dpctl.memory import MemoryUSMDevice, as_usm_memory
    from dpctl.tensor import usm_ndarray
    dpctl_available = True
except ImportError:
    dpctl_available = False


def _transfer_to_host(queue, *data):
    #TODO:
    host_data = []
    return queue, host_data


def _copy_to_usm(queue, array):
    if not dpctl_available:
        raise RuntimeError("dpctl need to be installed to work "
                           "with __sycl_usm_array_interface__")
    mem = MemoryUSMDevice(array.nbytes, queue=queue)
    mem.copy_from_host(array.tobytes())
    return usm_ndarray(array.shape, array.dtype, buffer=mem)


def _get_host_inputs(*args, **kwargs):
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return q, hostargs, hostkwargs


def _extract_usm_iface(*args, **kwargs):
    allargs = (*args, *kwargs.values())
    if len(allargs) == 0:
        return None
    return getattr(allargs[0],
                   '__sycl_usm_array_interface__',
                   None)


def _run_on_device(func, queue, obj=None, *args, **kwargs):
    # TODO:
    # queue.
    def dispatch_by_obj(obj, func, *args, **kwargs):
        if obj is not None:
            return func(obj, *args, **kwargs)
        return func(*args, **kwargs)

    return dispatch_by_obj(obj, func, *args, **kwargs)


def support_usm_ndarray(freefunc=False):
    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            usm_iface = _extract_usm_iface(*args, **kwargs)
            q, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            result = _run_on_device(func, q, obj, *hostargs, **hostkwargs)
            if usm_iface is not None and hasattr(result, '__array_interface__'):
                return _copy_to_usm(q, result)
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
