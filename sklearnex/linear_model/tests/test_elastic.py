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
    from sklearnex.linear_model import ElasticNet
    X, y = make_regression(n_features=2, random_state=0)
    elasticnet = ElasticNet(random_state=0).fit(X, y)
    assert 'daal4py' in elasticnet.__module__
    assert_allclose(elasticnet.intercept_, 1.451, atol=1e-3)
    assert_allclose(elasticnet.coef_, [18.838, 64.559], atol=1e-3)
