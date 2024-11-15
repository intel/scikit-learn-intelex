# ==============================================================================
# Copyright 2024 Intel Corporation
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

import numpy as np
import pytest

from onedal import _backend, _is_dpc_backend
from onedal.tests.utils._device_selection import get_queues
from onedal.utils._dpep_helpers import dpctl_available


@pytest.mark.skipif(
    not _is_dpc_backend or not dpctl_available, reason="requires dpc backend and dpctl"
)
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
@pytest.mark.parametrize("device_number", [None, 0, 1, 2, 3])
def test_sycl_queue_string_creation(device_type, device_number):
    # create devices from strings
    from dpctl import SyclQueue
    from dpctl._sycl_queue import SyclQueueCreationError

    onedal_SyclQueue = _backend.SyclQueue

    device = ":".join([device_type, str(device_number)]) if device_number else device_type

    raised_exception_dpctl = False
    raised_exception_backend = False

    try:
        dpctl_queue = SyclQueue(device)
    except SyclQueueCreationError:
        raised_exception_dpctl = True

    try:
        onedal_queue = onedal_SyclQueue(device)
    except RuntimeError:
        raised_exception_backend = True

    assert raised_exception_dpctl == raised_exception_backend
    # get_device_id must be modified to follow DPCtl conventions
    # this causes filter_string mismatches
    # if not raised_exception_backend:
    #    assert (
    #        onedal_queue.sycl_device.filter_string
    #        in dpctl_queue.sycl_device.filter_string
    #    )


@pytest.mark.skipif(
    not _is_dpc_backend or not dpctl_available, reason="requires dpc backend and dpctl"
)
@pytest.mark.parametrize("queue", get_queues())
def test_sycl_queue_conversion(queue):
    if queue is None:
        pytest.skip("Not a DPCtl queue")
    SyclQueue = queue.__class__
    onedal_SyclQueue = _backend.SyclQueue
    # convert back and forth to test `_get_capsule` attribute

    q = onedal_SyclQueue(queue)
    assert q.sycl_device.filter_string in queue.sycl_device.filter_string

    for i in range(10):
        q = SyclQueue(q.sycl_device.filter_string)
        q = onedal_SyclQueue(q)

    # verify the device is the same
    # get_device_id must be modified to follow DPCtl conventions
    # assert q.sycl_device.filter_string in queue.sycl_device.filter_string


@pytest.mark.skipif(
    not _is_dpc_backend or not dpctl_available, reason="requires dpc backend and dpctl"
)
@pytest.mark.parametrize("queue", get_queues())
def test_sycl_device_attributes(queue):
    from dpctl import SyclQueue

    if queue is None:
        pytest.skip("Not a DPCtl queue")
    onedal_SyclQueue = _backend.SyclQueue

    onedal_queue = onedal_SyclQueue(queue)

    # check fp64 support
    assert onedal_queue.sycl_device.has_aspect_fp64 == queue.sycl_device.has_aspect_fp64
    # check fp16 support
    assert onedal_queue.sycl_device.has_aspect_fp16 == queue.sycl_device.has_aspect_fp16
    # check is_cpu
    assert onedal_queue.sycl_device.is_cpu == queue.sycl_device.is_cpu
    # check is_gpu
    assert onedal_queue.sycl_device.is_gpu == queue.sycl_device.is_gpu
    # check device number
    # get_device_id must be modified to follow DPCtl conventions
    # assert onedal_queue.sycl_device.filter_string in queue.sycl_device.filter_string


@pytest.mark.skipif(not _is_dpc_backend, reason="requires dpc backend")
def test_backend_queue():
    q = _backend.SyclQueue("cpu")
    # verify copying via a py capsule object is functional
    q2 = _backend.SyclQueue(q._get_capsule())
    # verify copying via the _get_capsule attribute
    q3 = _backend.SyclQueue(q)

    q_array = [q, q2, q3]

    assert all([queue.sycl_device.has_aspect_fp64 for queue in q_array])
    assert all([queue.sycl_device.has_aspect_fp16 for queue in q_array])
    assert all([queue.sycl_device.is_cpu for queue in q_array])
    assert all([not queue.sycl_device.is_gpu for queue in q_array])
    assert all(["cpu" in queue.sycl_device.filter_string for queue in q_array])
