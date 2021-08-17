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


def _get_device_info_from_daal4py():
    import sys
    if 'daal4py.oneapi' in sys.modules:
        from daal4py.oneapi import _get_device_name_sycl_ctxt, _get_sycl_ctxt_params
        return _get_device_name_sycl_ctxt(), _get_sycl_ctxt_params()
    return None, dict()


def _get_global_queue():
    target = get_config()['target_offload']
    d4p_target, _ = _get_device_info_from_daal4py()

    if target != 'auto' or d4p_target != None:
        try:
            from dpctl import SyclQueue
        except ImportError:
            raise RuntimeError("dpctl need to be installed for device offload")

    if target != 'auto':
        if d4p_target != None:
            raise RuntimeError("Cannot use target offload option "
                               "inside daal4py.oneapi.sycl_context")
        if type(target) != SyclQueue:
            return SyclQueue(target)
        return target
    if d4p_target is not None and d4p_target != 'host':
        return SyclQueue(d4p_target)
    return None


def _transfer_to_host(queue, *data):
    has_usm_data = False

    host_data = []
    for item in data:
        usm_iface = getattr(item, '__sycl_usm_array_interface__', None)
        if usm_iface is not None:
            import dpctl.memory as dp_mem
            import numpy as np

            if queue is not None:
                if queue.sycl_device != usm_iface['syclobj'].sycl_device:
                    raise RuntimeError('Input data shall be located '
                                       'on single target device')
            else:
                queue = usm_iface['syclobj']

            buffer = dp_mem.as_usm_memory(item).copy_to_host()
            item = np.ndarray(shape=usm_iface['shape'], dtype=usm_iface['typestr'], buffer=buffer)
            has_usm_data = True
        elif has_usm_data and item is not None:
            raise RuntimeError('Input data shall be located '
                                'on single target device')
        host_data.append(item)
    return queue, host_data


def _copy_to_usm(queue, array):
    from dpctl.memory import MemoryUSMDevice
    from dpctl.tensor import usm_ndarray

    mem = MemoryUSMDevice(array.nbytes, queue=queue)
    mem.copy_from_host(array.tobytes())
    return usm_ndarray(array.shape, array.dtype, buffer=mem)


def wrap_output_data(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        data = (*args, *kwargs.values())
        usm_iface = getattr(data[0], '__sycl_usm_array_interface__', None)
        result = func(self, *args, **kwargs)
        if usm_iface is not None:
            return _copy_to_usm(usm_iface['syclobj'], result)
        return result
    return wrapper


def _check_supported(obj, queue, method_name, *data):
        if queue is not None:
            if queue.sycl_device.is_gpu and obj._gpu_supported(method_name, *data):
                return 'onedal', queue

        cpu_device = queue is None or queue.sycl_device.is_cpu

        _, d4p_options = _get_device_info_from_daal4py()
        allow_fallback = get_config()['allow_fallback_to_host'] or \
                         d4p_options.get('host_offload_on_fail', False)

        if cpu_device or allow_fallback:
            if obj._cpu_supported(method_name, *data):
                return 'onedal', queue if cpu_device else None
            else:
                return 'sklearn', None
        raise RuntimeError("Device support is not implemented")


def _dispatch(obj, method_name, branches, *args, **kwargs):
        import logging

        q = _get_global_queue()
        q, hostargs = _transfer_to_host(q, *args, **kwargs)
        backend, q = _check_supported(obj, q, method_name, *hostargs)

        logging.info(f"sklearn.{method_name}: {get_patch_message(backend)}")
        if backend == 'onedal':
            return branches[backend](q, *hostargs)
        if backend == 'sklearn':
            return branches[backend](*hostargs)
        raise RuntimeError(f'Undefined backend {backend} in {method_name}')

