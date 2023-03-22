#!/usr/bin/env python
#===============================================================================
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
#===============================================================================

import pytest

import numpy as np
from numpy.testing import assert_allclose
from daal4py.sklearn._utils import daal_check_version
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def test_sklearnex_import_rf_classifier():
    from sklearnex.preview.ensemble import RandomForestClassifier
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)
    assert 'sklearnex.preview' in rf.__module__
    assert_allclose([1], rf.predict([[0, 0, 0, 0]]))


def test_sklearnex_import_rf_regression():
    from sklearnex.preview.ensemble import RandomForestRegressor
    X, y = make_regression(n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    assert 'sklearnex.preview' in rf.__module__
    pred = rf.predict([[0, 0, 0, 0]])
    assert_allclose([-6.839], pred, atol=1e-2)


@pytest.mark.skipif(not daal_check_version((2023, 'P', 101)),
                    reason='requires OneDAL 2023.1.1')
def test_sklearnex_rf_classifier_splitter_mode():
    from sklearnex.preview.ensemble import RandomForestClassifier
    X, y = make_classification(n_samples=100, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=0)
    rf_b = RandomForestClassifier(max_depth=2,
                                  random_state=0,
                                  splitter_mode='best').fit(X_train, y_train)
    rf_r = RandomForestClassifier(max_depth=2,
                                  random_state=0,
                                  splitter_mode='random').fit(X_train, y_train)
    pred_b = rf_b.predict(X_test)
    pred_r = rf_r.predict(X_test)
    assert_allclose(pred_b, pred_r)


@pytest.mark.skipif(not daal_check_version((2023, 'P', 101)),
                    reason='requires OneDAL 2023.1.1')
def test_sklearnex_rf_regressor_splitter_mode():
    from sklearnex.preview.ensemble import RandomForestRegressor
    X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=0)
    rf_b = RandomForestRegressor(max_depth=2,
                                 random_state=0,
                                 splitter_mode='best').fit(X_train, y_train)
    rf_r = RandomForestRegressor(max_depth=2,
                                 random_state=0,
                                 splitter_mode='random').fit(X_train, y_train)
    pred_b = rf_b.predict(X_test)
    pred_r = rf_r.predict(X_test)
    assert_allclose(pred_b, pred_r)
