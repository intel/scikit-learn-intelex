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

import pytest
import numpy as np

from onedal.common._policy import _get_policy
from onedal.tests.utils._device_selection import (
    get_queues, get_memory_usm, is_dpctl_available, device_type_to_str)


@pytest.mark.parametrize('queue', get_queues())
def test_queue_passed_directly(queue):
    device_name = device_type_to_str(queue)
    assert _get_policy(queue).get_device_name() == device_name


@pytest.mark.parametrize('queue', get_queues())
def test_with_numpy_data(queue):
    X = np.zeros((5, 3))
    y = np.zeros(3)

    device_name = device_type_to_str(queue)
    assert _get_policy(queue, X, y).get_device_name() == device_name


@pytest.mark.skipif(not is_dpctl_available(), reason='depends on dpctl')
@pytest.mark.parametrize('queue', get_queues('cpu,gpu'))
@pytest.mark.parametrize('memtype', get_memory_usm())
def test_with_usm_ndarray_data(queue, memtype):
    from dpctl.tensor import usm_ndarray

    device_name = device_type_to_str(queue)
    X = usm_ndarray((5, 3), buffer=memtype(5 * 3 * 8, queue=queue))
    y = usm_ndarray((3, ), buffer=memtype(3 * 8, queue=queue))
    assert _get_policy(None, X, y).get_device_name() == device_name


@pytest.mark.skipif(not is_dpctl_available(['cpu', 'gpu']),
                    reason='test uses multiple devices')
@pytest.mark.parametrize('memtype', get_memory_usm())
def test_queue_parameter_with_usm_ndarray(memtype):
    from dpctl import SyclQueue
    from dpctl.tensor import usm_ndarray

    q1 = SyclQueue('cpu')
    q2 = SyclQueue('gpu')

    X = usm_ndarray((5, 3), buffer=memtype(5 * 3 * 8, queue=q1))
    assert _get_policy(q2, X).get_device_name() == device_type_to_str(q2)
