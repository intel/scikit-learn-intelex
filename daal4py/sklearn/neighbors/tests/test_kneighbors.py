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
import numpy as np

from daal4py.sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

DISTANCES = ['minkowski']
ALGORITHMS = ['brute', 'kd_tree', 'auto']
WEIGHTS = ['uniform', 'distance']
KS = [1, 3, 7, 25]


def make_dataset(n_samples=256, n_features=5, n_classes=2, test_size=0.5, shuffle=True):
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=777)
    return train_test_split(x, y, random_state=777, test_size=test_size, shuffle=shuffle)

@pytest.mark.parametrize('distance', DISTANCES)
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('weight', WEIGHTS)
@pytest.mark.parametrize('k', KS)
def test_check_determenistic(distance, algorithm, weight, k):
    check_determenistic(distance, algorithm, weight, k)

def check_determenistic(distance, algorithm, weight, k):
    N_TRIES = 5

    x_train, x_test, y_train, y_test = make_dataset()

    alg_results = []
    for _ in range(N_TRIES):
        alg = KNeighborsClassifier(
            n_neighbors=k, weights=weight, algorithm=algorithm,
            leaf_size=30, p=2, metric=distance)
        alg.fit(x_train, y_train)
        distances, indices = alg.kneighbors(x_test)
        labels = alg.predict(x_test)
        alg_results.append((distances, indices, labels))
        print((y_test == labels).mean())

    for i in range(1, N_TRIES):
        for j, res in enumerate(alg_results[i]):
            if (res == alg_results[0][j]).mean() != 1:
                raise ValueError('Results are different between runs for {} {} {} k={}'.format(
                    algorithm, weight, distance, k))

def convert_data(data, class_name=np.array, order='C', dtype=np.float64):
    if order == 'C':
        data = np.ascontiguousarray(data, dtype=dtype)
    else:
        data = np.asfortranarray(data, dtype=dtype)
    return class_name(data)

@pytest.mark.parametrize('distance', DISTANCES)
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('weight', WEIGHTS)
@pytest.mark.parametrize('k', KS)
def test_data_formats_diff(distance, algorithm, weight, k):
    check_data_formats_diff(distance, algorithm, weight, k)

def check_data_formats_diff(distance, algorithm, weight, k):
    pd = pytest.importorskip('pandas')

    data_formats = [pd.DataFrame, np.array]
    orders = ['C', 'F']

    x_train, x_test, y_train, y_test = make_dataset()

    alg_results = []
    for data_format in data_formats:
        for order in orders:
            x_train_copy = convert_data(x_train.copy(), data_format, order)
            x_test_copy = convert_data(x_test.copy(), data_format, order)
            y_train_copy = convert_data(y_train.copy(), data_format, order)
            y_test_copy = convert_data(y_test.copy(), data_format, order)

            alg = KNeighborsClassifier(
            n_neighbors=k, weights=weight, algorithm=algorithm,
            leaf_size=30, p=2, metric=distance)
            alg.fit(x_train_copy, y_train_copy)
            distances, indices = alg.kneighbors(x_test_copy)
            labels = alg.predict(x_test_copy)
            alg_results.append((distances, indices, labels))

    for i in range(1, len(alg_results)):
        for j, res in enumerate(alg_results[i]):
            if (res == alg_results[0][j]).mean() != 1:
                raise ValueError('Results are different between formats')
