#===============================================================================
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
#===============================================================================

from ._config import get_config
from ._utils import get_patch_message
from functools import wraps
import numpy as np

try:
    from dpctl import SyclQueue
    from dpctl.memory import MemoryUSMDevice, as_usm_memory
    from dpctl.tensor import usm_ndarray
    dpctl_available = True
except ImportError:
    dpctl_available = False


class DummySyclQueue:
    '''This class is designed to act like dpctl.SyclQueue
    to allow device dispatching in scenarios when dpctl is not available'''

    class DummySyclDevice:
        def __init__(self, filter_string):
            self._filter_string = filter_string
            self.is_cpu = 'cpu' in filter_string
            self.is_gpu = 'gpu' in filter_string
            self.is_host = False

            if not (self.is_cpu or self.is_host):
                import logging
                logging.warning("Device support is limited. "
                                "Please install dpctl for full experience")

        def get_filter_string(self):
            return self._filter_string

    def __init__(self, filter_string):
        self.sycl_device = self.DummySyclDevice(filter_string)


def _get_device_info_from_daal4py():
    import sys
    if 'daal4py.oneapi' in sys.modules:
        from daal4py.oneapi import _get_device_name_sycl_ctxt, _get_sycl_ctxt_params
        return _get_device_name_sycl_ctxt(), _get_sycl_ctxt_params()
    return None, dict()


def _get_global_queue():
    target = get_config()['target_offload']
    d4p_target, _ = _get_device_info_from_daal4py()
    if d4p_target == 'host':
        d4p_target = 'cpu'

    QueueClass = DummySyclQueue if not dpctl_available else SyclQueue

    if target != 'auto':
        if d4p_target is not None and \
           d4p_target != target:
            if not isinstance(target, str):
                if d4p_target not in target.sycl_device.get_filter_string():
                    raise RuntimeError("Cannot use target offload option "
                                       "inside daal4py.oneapi.sycl_context")
            else:
                raise RuntimeError("Cannot use target offload option "
                                   "inside daal4py.oneapi.sycl_context")
        if isinstance(target, QueueClass):
            return target
        return QueueClass(target)
    if d4p_target is not None:
        return QueueClass(d4p_target)
    return None


def _transfer_to_host(queue, *data):
    has_usm_data, has_host_data = False, False

    host_data = []
    for item in data:
        usm_iface = getattr(item, '__sycl_usm_array_interface__', None)
        if usm_iface is not None:
            if not dpctl_available:
                raise RuntimeError("dpctl need to be installed to work "
                                   "with __sycl_usm_array_interface__")
            if queue is not None:
                if queue.sycl_device != usm_iface['syclobj'].sycl_device:
                    raise RuntimeError('Input data shall be located '
                                       'on single target device')
            else:
                queue = usm_iface['syclobj']

            buffer = as_usm_memory(item).copy_to_host()
            item = np.ndarray(shape=usm_iface['shape'],
                              dtype=usm_iface['typestr'],
                              buffer=buffer)
            has_usm_data = True
        else:
            has_host_data = True

        mismatch_host_item = usm_iface is None and item is not None and has_usm_data
        mismatch_usm_item = usm_iface is not None and has_host_data

        if mismatch_host_item or mismatch_usm_item:
            raise RuntimeError('Input data shall be located on single target device')

        host_data.append(item)
    return queue, host_data


def _get_backend(obj, queue, method_name, *data):
    cpu_device = queue is None or queue.sycl_device.is_cpu
    gpu_device = queue is not None and queue.sycl_device.is_gpu
    cpu_fallback = False

    if (cpu_device and obj._onedal_cpu_supported(method_name, *data)) or \
       (gpu_device and obj._onedal_gpu_supported(method_name, *data)):
        return 'onedal', queue, cpu_fallback
    if cpu_device:
        return 'sklearn', None, cpu_fallback

    _, d4p_options = _get_device_info_from_daal4py()
    allow_fallback = get_config()['allow_fallback_to_host'] or \
        d4p_options.get('host_offload_on_fail', False)

    if gpu_device and allow_fallback:
        if obj._onedal_cpu_supported(method_name, *data):
            cpu_fallback = True
            return 'onedal', None, cpu_fallback
        return 'sklearn', None, cpu_fallback

    raise RuntimeError("Device support is not implemented")


def dispatch(obj, method_name, branches, *args, **kwargs):
    import logging

    q = _get_global_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))

    backend, q, cpu_fallback = _get_backend(obj, q, method_name, *hostargs)

    logging.info(f"sklearn.{method_name}: {get_patch_message(backend, q, cpu_fallback)}")
    if backend == 'onedal':
        return branches[backend](obj, *hostargs, **hostkwargs, queue=q)
    if backend == 'sklearn':
        return branches[backend](obj, *hostargs, **hostkwargs)
    raise RuntimeError(f'Undefined backend {backend} in {method_name}')


def _copy_to_usm(queue, array):
    if not dpctl_available:
        raise RuntimeError("dpctl need to be installed to work "
                           "with __sycl_usm_array_interface__")
    mem = MemoryUSMDevice(array.nbytes, queue=queue)
    mem.copy_from_host(array.tobytes())
    return usm_ndarray(array.shape, array.dtype, buffer=mem)


def wrap_output_data(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        data = (*args, *kwargs.values())
        if len(data) == 0:
            usm_iface = None
        else:
            usm_iface = getattr(data[0], '__sycl_usm_array_interface__', None)
        result = func(self, *args, **kwargs)
        if usm_iface is not None:
            return _copy_to_usm(usm_iface['syclobj'], result)
        return result
    return wrapper
