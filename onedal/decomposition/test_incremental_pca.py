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

import pytest
from onedal.decomposition import IncrementalPCA
from onedal.tests.utils._device_selection import get_queues
import numpy as np


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_on_gold_data(queue, dtype):
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, 2)
    incpca = IncrementalPCA()

    for i in range(2):
        incpca.partial_fit(X_split[i], queue=queue)

    result = incpca.finalize_fit()
    
