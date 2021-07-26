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

from onedal import _backend


def _get_policy(queue, *data):
    data_queue = _get_queue(*data)

    if queue is None:
        if data_queue is None:
            return _HostInteropPolicy()
        return _DataParallelInteropPolicy(data_queue)
    return _DataParallelInteropPolicy(queue)


def _get_queue(*data):
    if len(data) > 0 and hasattr(data[0], '__sycl_usm_array_interface__'):
        return data[0].__sycl_usm_array_interface__['syclobj']
    return None
    # queues = [
    #     item.__sycl_usm_array_interface__['syclobj']
    #     for item in data if hasattr(item, '__sycl_usm_array_interface__') and item is not None
    # ]

    # for i in range(0, len(queues) - 1):
    #     if queues[i].sycl_device != queues[i].sycl_device:
    #         raise RuntimeError("All input data object shall reside on the same device")

    # return queues[0] if len(queues) > 0 else None


def _get_device_name_from_daal4py():
    import sys
    if 'daal4py.oneapi' in sys.modules:
        import daal4py.oneapi as d4p_oneapi
        return d4p_oneapi._get_device_name_sycl_ctxt()
    return None


class _Daal4PyContextSaver:
    def __init__(self):
        import sys

        self._d4p_context = None
        self._host_context = None
        if 'daal4py.oneapi' is sys.modules:
            from daal4py.oneapi import _get_sycl_ctxt, sycl_execution_context
            self._d4p_context = _get_sycl_ctxt()
            self._host_context = sycl_execution_context('host')
            self._host_context.apply()

    def __del__(self):
        if self._d4p_context:
            self._d4p_context.apply()


class _DataParallelInteropPolicy(_backend.data_parallel_policy):
    def __init__(self, queue):
        self._queue = queue
        self._d4p_interop = _Daal4PyContextSaver()
        super().__init__(self._queue.addressof_ref())


class _HostInteropPolicy(_backend.host_policy):
    def __init__(self):
        super().__init__()
        self._d4p_interop = _Daal4PyContextSaver()

