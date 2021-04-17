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
    from sklearnex.linear_model import Lasso
    X = [[0,0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    lasso = Lasso(alpha=0.1).fit(X, y)
    assert 'daal4py' in lasso.__module__
    assert_allclose(lasso.intercept_, 0.15)
    assert_allclose(lasso.coef_, [0.85, 0.  ])
