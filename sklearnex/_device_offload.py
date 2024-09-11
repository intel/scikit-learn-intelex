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

from functools import wraps

from onedal._device_offload import (
    _copy_to_usm,
    _get_global_queue,
    _transfer_to_host,
    dpnp_available,
)

if dpnp_available:
    import dpnp
    from onedal._device_offload import _convert_to_dpnp


from ._config import get_config


def _get_backend(obj, queue, method_name, *data):
    cpu_device = queue is None or queue.sycl_device.is_cpu
    gpu_device = queue is not None and queue.sycl_device.is_gpu

    if cpu_device:
        patching_status = obj._onedal_cpu_supported(method_name, *data)
        if patching_status.get_status():
            return "onedal", queue, patching_status
        else:
            return "sklearn", None, patching_status

    allow_fallback_to_host = get_config()["allow_fallback_to_host"]

    if gpu_device:
        patching_status = obj._onedal_gpu_supported(method_name, *data)
        if patching_status.get_status():
            return "onedal", queue, patching_status
        else:
            if allow_fallback_to_host:
                patching_status = obj._onedal_cpu_supported(method_name, *data)
                if patching_status.get_status():
                    return "onedal", None, patching_status
                else:
                    return "sklearn", None, patching_status
            else:
                return "sklearn", None, patching_status

    raise RuntimeError("Device support is not implemented")


def dispatch(obj, method_name, branches, *args, **kwargs):
    q = _get_global_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))

    backend, q, patching_status = _get_backend(obj, q, method_name, *hostargs)

    if backend == "onedal":
        patching_status.write_log(queue=q)
        return branches[backend](obj, *hostargs, **hostkwargs, queue=q)
    if backend == "sklearn":
        patching_status.write_log()
        return branches[backend](obj, *hostargs, **hostkwargs)
    raise RuntimeError(
        f"Undefined backend {backend} in " f"{obj.__class__.__name__}.{method_name}"
    )


def wrap_output_data(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        data = (*args, *kwargs.values())
        if len(data) == 0:
            usm_iface = None
        else:
            usm_iface = getattr(data[0], "__sycl_usm_array_interface__", None)
        result = func(self, *args, **kwargs)
        if usm_iface is not None:
            result = _copy_to_usm(usm_iface["syclobj"], result)
            if dpnp_available and isinstance(data[0], dpnp.ndarray):
                result = _convert_to_dpnp(result)
        return result

    return wrapper
