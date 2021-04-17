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


def test_sklearnex_import_svc():
    from sklearnex.svm import SVC
    X = np.array([[-2, -1], [-1, -1], [-1, -2],
                  [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    svc = SVC(kernel='linear').fit(X, y)
    assert 'daal4py' in svc.__module__
    assert_allclose(svc.dual_coef_, [[-0.25, .25]])
    assert_allclose(svc.support_, [1, 3])
