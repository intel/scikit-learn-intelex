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

try:
    from sklearnex._config import get_config
    from sklearnex._device_offload import (_get_global_queue,
                                           _transfer_to_host,
                                           _copy_to_usm)
    _sklearnex_available = True
except ImportError:
    import logging
    logging.warning('Device support is limited in daal4py patching. '
                    'Use Intel(R) Extension for Scikit-learn* '
                    'for full experience.')
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
    return getattr(allargs[0],
                   '__sycl_usm_array_interface__',
                   None)


def _run_on_device(func, queue, obj=None, *args, **kwargs):
    def dispatch_by_obj(obj, func, *args, **kwargs):
        if obj is not None:
            return func(obj, *args, **kwargs)
        return func(*args, **kwargs)

    if queue is not None:
        from daal4py.oneapi import sycl_context, _get_in_sycl_ctxt

        if _get_in_sycl_ctxt() is False:
            host_offload = get_config()['allow_fallback_to_host']

            with sycl_context('gpu' if queue.sycl_device.is_gpu else 'cpu',
                              host_offload_on_fail=host_offload):
                return dispatch_by_obj(obj, func, *args, **kwargs)
    return dispatch_by_obj(obj, func, *args, **kwargs)


def support_usm_ndarray(freefunc=False):
    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            if _sklearnex_available:
                usm_iface = _extract_usm_iface(*args, **kwargs)
                q, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
                result = _run_on_device(func, q, obj, *hostargs, **hostkwargs)
                if usm_iface is not None and hasattr(result, '__array_interface__'):
                    return _copy_to_usm(q, result)
                return result
            return _run_on_device(func, None, obj, *args, **kwargs)

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
