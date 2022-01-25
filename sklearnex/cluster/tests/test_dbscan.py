#!/usr/bin/env python
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

import numpy as np
from numpy.testing import assert_allclose


def test_sklearnex_import():
    from sklearnex.cluster import DBSCAN
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [25, 80]])
    dbscan = DBSCAN(eps=3, min_samples=2).fit(X)
    assert 'daal4py' in dbscan.__module__

    result = dbscan.labels_
    expected = np.array([0, 0, 0, 1, 1, -1], dtype=np.int32)
    assert_allclose(expected, result)
