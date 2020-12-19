#
# ****************************************************************************
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
# ****************************************************************************

import pandas as pd
import pytest
import numpy as np
from daal4py.sklearn.ensemble \
    import RandomForestClassifier as DaalRandomForestClassifier
from daal4py.sklearn.ensemble \
    import RandomForestRegressor as DaalRandomForestRegressor
from daal4py.sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DISTANCES = ['minkowski']
ALGORITHMS = ['brute', 'kd_tree', 'auto']
WEIGHTS = ['uniform', 'distance']
KS = [1, 3, 7, 25]
N_TRIES = 10
CHECK_RATIO_KNN = 0.85


def make_dataset(n_samples=256, n_features=5, n_classes=2,
                 test_size=0.5, shuffle=True):
    x, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, random_state=777)
    return train_test_split(x, y, random_state=777,
                            test_size=test_size, shuffle=shuffle)


def check_determenistic(distance, algorithm, weight, k):
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
        accuracy = accuracy_score(labels, y_test)
        assert accuracy >= CHECK_RATIO_KNN,\
            'kNN classifier:accuracy={}'.format(accuracy)

    for i in range(1, N_TRIES):
        for j, res in enumerate(alg_results[i]):
            assert (res == alg_results[0][j]).mean() == 1, \
                ('Results are different between runs for %s, %s, %s, k=%d'\
                % (algorithm, weight, distance, k))


@pytest.mark.parametrize('distance', DISTANCES)
@pytest.mark.parametrize('algorithm', ALGORITHMS)
@pytest.mark.parametrize('weight', WEIGHTS)
@pytest.mark.parametrize('k', KS)
def test_check_determenistic(distance, algorithm, weight, k):
    check_determenistic(distance, algorithm, weight, k)
