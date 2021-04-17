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
from sklearn.datasets import make_regression

def test_sklearnex_import():
    from sklearnex.linear_model import LinearRegression
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    linreg = LinearRegression().fit(X, y)
    assert 'daal4py' in linreg.__module__
    assert_allclose(linreg.intercept_, 3.)
    assert_allclose(linreg.coef_, [1., 2.])
