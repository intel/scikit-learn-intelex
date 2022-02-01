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
from sklearn.datasets import make_classification, make_regression
from daal4py.sklearn._utils import daal_check_version


def test_sklearnex_import_rf_classifier():
    from sklearnex.ensemble import RandomForestClassifier
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)
    assert 'daal4py' in rf.__module__
    assert_allclose([1], rf.predict([[0, 0, 0, 0]]))


def test_sklearnex_import_rf_regression():
    from sklearnex.ensemble import RandomForestRegressor
    X, y = make_regression(n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    assert 'daal4py' in rf.__module__
    if daal_check_version((2021, 'P', 400)):
        assert_allclose([-6.97], rf.predict([[0, 0, 0, 0]]), atol=1e-2)
    else:
        assert_allclose([-6.66], rf.predict([[0, 0, 0, 0]]), atol=1e-2)
