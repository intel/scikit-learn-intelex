#===============================================================================
# Copyright 2020-2021 Intel Corporation
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

import pandas as pd
import pytest
import numpy as np
from daal4py.sklearn.ensemble \
    import RandomForestClassifier as DaalRandomForestClassifier
from daal4py.sklearn.ensemble \
    import RandomForestRegressor as DaalRandomForestRegressor
from daal4py.sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def convert_data(data, class_name=np.array, order='C', dtype=np.float64):
    if order == 'C':
        data = np.ascontiguousarray(data, dtype=dtype)
    else:
        data = np.asfortranarray(data, dtype=dtype)
    return class_name(data)


def make_dataset(n_samples=256, n_features=5, n_classes=2,
                 test_size=0.5, shuffle=True):
    x, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes, random_state=777)
    return train_test_split(x, y, random_state=777,
                            test_size=test_size, shuffle=shuffle)


ESTIMATORS = {
    'KNeighborsClassifier':
        KNeighborsClassifier(n_neighbors=10),
    'DaalRandomForestClassifier':
        DaalRandomForestClassifier(n_estimators=10, random_state=777),
    'DaalRandomForestRegressor':
        DaalRandomForestRegressor(n_estimators=10, random_state=777),
}

ORDERS = ['C', 'F']
DATA_FORMATS = [pd.DataFrame, np.array]


def check_data_formats_diff(name):
    x_train, x_test, y_train, y_test = make_dataset()
    alg_results = []
    for data_format in DATA_FORMATS:
        for order in ORDERS:
            x_train_copy = convert_data(x_train.copy(), data_format, order)
            x_test_copy = convert_data(x_test.copy(), data_format, order)
            y_train_copy = convert_data(y_train.copy(), data_format, order)
            alg = ESTIMATORS[name]
            alg.fit(x_train_copy, y_train_copy)
            labels = alg.predict(x_test_copy)
            alg_results.append(labels)

    for i in range(1, len(alg_results)):
        for j, res in enumerate(alg_results[i]):
            assert (res == alg_results[0][j]).mean() == 1, \
                ('Results are different between formats: estimator=%s' % (name))


@pytest.mark.parametrize('name', ESTIMATORS)
def test_data_formats_diff(name):
    check_data_formats_diff(name)
