# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
def test_onedal_import_covariance(queue):
    from onedal.covariance import IncrementalEmpiricalCovariance

    X = np.array([[0, 1], [0, 1]])
    X_split = np.array_split(X, 2)
    inccov = IncrementalEmpiricalCovariance()
    for i in range(2):
        inccov.partial_compute(X_split[i], queue=queue)
    
    result = inccov.finalize_compute()
    expected_covariance = np.array([[0, 0], [0, 0]])
    expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)