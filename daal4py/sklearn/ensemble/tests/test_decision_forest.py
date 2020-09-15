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

import pytest
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestClassifier as D4PRandomForestClassifier

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
def test_class_weight_iris(weight):
    check_class_weight_iris(weight)

def check_class_weight_iris(weight, check_ratio=0.9):
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

def make_filled_list(list_size, fill):
    return [fill for i in range(list_size)]

SAMPLE_WEIGHTS_IRIS = [
    (make_filled_list(100, 0), '0'),
    (make_filled_list(100, 1), '1'),
    (make_filled_list(100, 5), '5'),
    (make_filled_list(100, 50), '50'),
    (make_filled_list(100, random.uniform(1, 50)), 'Random [1, 50]'),
    (make_filled_list(100, random.uniform(1, 100)), 'Random [1, 100]'),
    (make_filled_list(100, random.uniform(1, 500)), 'Random [1, 500]'),
    (make_filled_list(100, random.uniform(1, 1000)), 'Random [1, 1000]'),
    (make_filled_list(100, random.uniform(1, 5000)), 'Random [1, 5000]'),
]

@pytest.mark.parametrize('weight', SAMPLE_WEIGHTS_IRIS)
def test_sample_weight_iris(weight):
    check_sample_weight(weight)

def check_sample_weight(weight, check_ratio=0.9):
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
    