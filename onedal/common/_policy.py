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

import sys

from onedal import _backend, _is_dpc_backend


def _get_policy(queue, *data):
    data_queue = _get_queue(*data)
    if _is_dpc_backend:
        if queue is None:
            if data_queue is None:
                return _HostInteropPolicy()
            return _DataParallelInteropPolicy(data_queue)
        return _DataParallelInteropPolicy(queue)
    else:
        if not (data_queue is None and queue is None):
            raise RuntimeError(
                "Operation using the requested SYCL queue requires the DPC backend"
            )
        return _HostInteropPolicy()


def _get_queue(*data):
    if len(data) > 0 and hasattr(data[0], "__sycl_usm_array_interface__"):
        # Assume that all data reside on the same device
        return data[0].__sycl_usm_array_interface__["syclobj"]
    return None


class _HostInteropPolicy(_backend.host_policy):
    def __init__(self):
        super().__init__()


if _is_dpc_backend:
    from onedal._device_offload import DummySyclQueue

    class _DataParallelInteropPolicy(_backend.data_parallel_policy):
        def __init__(self, queue):
            self._queue = queue
            if isinstance(queue, DummySyclQueue):
                super().__init__(self._queue.sycl_device.get_filter_string())
                return
            super().__init__(self._queue)
