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

import unittest
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestClassifier as D4PRandomForestClassifier

class Test(unittest.TestCase):

    def make_dataset(n_samples, n_features, random_state=37):
        x, y = make_classification(n_samples, n_features, random_state=random_state)
        return x, y

    def test_class_weight(self):
        X, y = load_iris(return_X_y =True)
        X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)
        weights = [
            {0: 0, 1: 0, 2: 0},
            {0: 0, 1: 1, 2: 1},
            {0: 1, 1: 2, 2: 3},
            {0: 10, 1: 5, 2: 4},
            {0: random.uniform(1, 50), 1: random.uniform(1, 50), 2: random.uniform(1, 50)},
            {0: random.uniform(50, 100), 1: random.uniform(50, 100), 2: random.uniform(50, 100)},
            {0: random.uniform(1, 1000), 1: random.uniform(1, 1000), 2: random.uniform(1, 1000)},
            {0: 50, 1: 50, 2: 50},
            'balanced',
        ]

        for weight in weights:
            SK_model = SKRandomForestClassifier(class_weight=weight)
            D4P_model = D4PRandomForestClassifier(class_weight=weight)

            SK_predict = SK_model.fit(X_train, y_train).predict(x_test)
            D4P_predict = D4P_model.fit(X_train, y_train).predict(x_test)

            SK_accuracy = accuracy_score(SK_predict, y_test)
            D4P_accuracy = accuracy_score(D4P_predict, y_test)
            ratio = D4P_accuracy / SK_accuracy

            assert ratio >= 0.9, 'Failed in testing class weights, weight = ' + str(weight) + ', Accuracy ratio = ' + str(ratio)

    def make_filled_list(list_size, fill):
        return [fill for i in range(list_size)]

    def test_sample_weight(self):
        X, y = load_iris(return_X_y =True)
        X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=31)
        weights = [
            (Test.make_filled_list(X_train.shape[0], 0), '0'),
            (Test.make_filled_list(X_train.shape[0], 1), '1'),
            (Test.make_filled_list(X_train.shape[0], 5), '5'),
            (Test.make_filled_list(X_train.shape[0], 50), '50'),
            (Test.make_filled_list(X_train.shape[0], random.uniform(1, 5000)), 'Random'),
            (Test.make_filled_list(X_train.shape[0], random.uniform(1, 5000)), 'Random'),
            (Test.make_filled_list(X_train.shape[0], random.uniform(1, 5000)), 'Random'),
            (Test.make_filled_list(X_train.shape[0], random.uniform(1, 5000)), 'Random'),
            (Test.make_filled_list(X_train.shape[0], random.uniform(1, 5000)), 'Random'),
        ]
        for weight in weights:
            SK_model = SKRandomForestClassifier()
            D4P_model = D4PRandomForestClassifier()

            SK_predict = SK_model.fit(X_train, y_train, sample_weight=weight[0]).predict(x_test)
            D4P_predict = D4P_model.fit(X_train, y_train, sample_weight=weight[0]).predict(x_test)

            SK_accuracy = accuracy_score(SK_predict, y_test)
            D4P_accuracy = accuracy_score(D4P_predict, y_test)
            ratio = D4P_accuracy / SK_accuracy

            assert ratio >= 0.9, 'Failed in testing sample weights, list filled ' + weight[1] + ', Accuracy ratio = ' + str(ratio)

if __name__ == '__main__':
    unittest.main()
