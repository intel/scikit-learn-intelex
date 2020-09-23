#
#*******************************************************************************
# Copyright 2020 Intel Corporation
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
#******************************************************************************/

from daal4py import __daal_run_version__, __daal_link_version__

daal_run_version = tuple(map(int, (__daal_run_version__[0:4], __daal_run_version__[4:8])))
daal_link_version = tuple(map(int, (__daal_link_version__[0:4], __daal_link_version__[4:8])))

import pytest
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestClassifier as D4PRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SKRandomForestRegressor
from daal4py.sklearn.ensemble import RandomForestRegressor as D4PRandomForestRegressor

CLASS_WEIGHTS_IRIS = [
    {0: 0, 1: 0, 2: 0},
    {0: 0, 1: 1, 2: 1},
    {0: 1, 1: 2, 2: 3},
    {0: 10, 1: 5, 2: 4},
    {0: random.uniform(1, 50), 1: random.uniform(1, 50), 2: random.uniform(1, 50)},
    {0: random.uniform(50, 100), 1: random.uniform(50, 100), 2: random.uniform(50, 100)},
    {0: random.uniform(1, 1000), 1: random.uniform(1, 1000), 2: random.uniform(1, 1000)},
    {0: random.uniform(1, 10), 1: random.uniform(50, 100), 2: random.uniform(1, 100)},
    {0: random.uniform(1, 10), 1: random.uniform(1, 100), 2: random.uniform(1, 1000)},
    {0: random.uniform(1, 2000), 1: random.uniform(1, 2000), 2: random.uniform(1, 2000)},
    {0: 50, 1: 50, 2: 50},
    'balanced',
]

@pytest.mark.parametrize('weight', CLASS_WEIGHTS_IRIS)
def test_classifier_class_weight_iris(weight):
    if daal_run_version >= (2020, 3) and daal_link_version >= (2020, 3):
        check_classifier_class_weight_iris(weight)

def check_classifier_class_weight_iris(weight, check_ratio=0.85):
    X, y = load_iris(return_X_y =True)
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)  
    
    SK_model = SKRandomForestClassifier(class_weight=weight)
    D4P_model = D4PRandomForestClassifier(class_weight=weight)

    SK_predict = SK_model.fit(X_train, y_train).predict(x_test)
    D4P_predict = D4P_model.fit(X_train, y_train).predict(x_test)

    SK_accuracy = accuracy_score(SK_predict, y_test)
    D4P_accuracy = accuracy_score(D4P_predict, y_test)
    ratio = D4P_accuracy / SK_accuracy
    assert ratio >= check_ratio

SAMPLE_WEIGHTS_IRIS = [
    (np.full_like(range(100), 0), 'Only 0'),
    (np.full_like(range(100), 1), 'Only 1'),
    (np.full_like(range(100), 5), 'Only 5'),
    (np.full_like(range(100), 50), 'Only 50'),
    (np.random.rand(100), 'Uniform distribution'),
    (np.random.normal(1000, 10, 100), 'Gaussian distribution'),
    (np.random.exponential(5, 100), 'Exponential distribution'),
    (np.random.poisson(lam=10, size=100), 'Poisson distribution'),
    (np.random.rayleigh(scale=1, size=100), 'Rayleigh distribution'),
]

@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_classifier_sample_weight_iris(weight):
    if daal_run_version >= (2020, 3) and daal_link_version >= (2020, 3):
        check_classifier_sample_weight(weight)

def check_classifier_sample_weight(weight, check_ratio=0.9):
    X, y = load_iris(return_X_y =True)
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)
        
    SK_model = SKRandomForestClassifier()
    D4P_model = D4PRandomForestClassifier()

    SK_predict = SK_model.fit(X_train, y_train, sample_weight=weight[0]).predict(x_test)
    D4P_predict = D4P_model.fit(X_train, y_train, sample_weight=weight[0]).predict(x_test)

    SK_accuracy = accuracy_score(SK_predict, y_test)
    D4P_accuracy = accuracy_score(D4P_predict, y_test)
    ratio = D4P_accuracy / SK_accuracy
    assert ratio >= check_ratio, 'Failed testing sample weights, sample_weight_type = ' + weight[1]

@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_regressor_sample_weight_iris(weight):
    if daal_run_version >= (2020, 3) and daal_link_version >= (2020, 3) and weight[1] != 'Only 0':
        check_regressor_sample_weight(weight)

def check_regressor_sample_weight(weight, check_ratio=1.4):
    X, y = load_iris(return_X_y =True)
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)
        
    SK_model = SKRandomForestRegressor()
    D4P_model = D4PRandomForestRegressor()

    SK_predict = SK_model.fit(X_train, y_train, sample_weight=weight[0]).predict(x_test)
    D4P_predict = D4P_model.fit(X_train, y_train, sample_weight=weight[0]).predict(x_test)

    SK_accuracy = mean_squared_error(SK_predict, y_test)
    D4P_accuracy = mean_squared_error(D4P_predict, y_test)
    ratio = D4P_accuracy / SK_accuracy
    assert ratio <= check_ratio, 'Failed while testing regression sample weights, sample_weight_type = ' + weight[1]
