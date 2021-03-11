# ===============================================================================
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
# ===============================================================================

import pytest
import types
import tracemalloc
from daal4py.sklearn.monkeypatch.dispatcher import _get_map_of_algorithms
from daal4py.sklearn.model_selection import _daal_train_test_split
from daal4py.sklearn.utils.validation import _daal_assert_all_finite
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification, make_regression
import pandas as pd
import numpy as np


class TrainTestSplitEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        _daal_train_test_split(x, y)


class AssertAllFiniteEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        _daal_assert_all_finite(x)
        _daal_assert_all_finite(y)


ESTIMATORS = [
    TrainTestSplitEstimator,
    AssertAllFiniteEstimator
]

BANNED_ESTIMATORS = [
    'TSNE',  # too slow for using in testing on applicable data sizes
    'LogisticRegression'  # disabled because of known fails
]

# add all daa4lpy estimators enabled in patching (except banned)
patched_estimators = _get_map_of_algorithms().values()
for listing in patched_estimators:
    estimator, name = listing[0][0][2], listing[0][0][1]
    if not isinstance(estimator, types.FunctionType):
        if name not in BANNED_ESTIMATORS:
            if isinstance(estimator(), BaseEstimator):
                if hasattr(estimator, 'fit'):
                    ESTIMATORS.append(estimator)


def ndarray_c(x, y):
    return np.ascontiguousarray(x), y


def ndarray_f(x, y):
    return np.asfortranarray(x), y


def dataframe_c(x, y):
    return pd.DataFrame(x), pd.Series(y)


def dataframe_f(x, y):
    return pd.DataFrame(np.asfortranarray(x)), pd.Series(y)


DATA_TRANSFORMS = [
    ndarray_c,
    ndarray_f,
    dataframe_c,
    dataframe_f
]


def gen_clsf_data():
    data, label = make_classification(
        n_samples=2000, n_features=50, random_state=777)
    return data, label, \
        data.size * data.dtype.itemsize + label.size * label.dtype.itemsize


def _kfold_function_template(estimator, data_transform_function):
    tracemalloc.start()

    x, y, data_memory_size = gen_clsf_data()
    kf = KFold(n_splits=10)
    x, y = data_transform_function(x, y)

    mem_before, _ = tracemalloc.get_traced_memory()
    for train_index, test_index in kf.split(x):
        if isinstance(x, np.ndarray):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
        elif isinstance(x, pd.core.frame.DataFrame):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        alg = estimator()
        alg.fit(x_train, y_train)
        if hasattr(alg, 'predict'):
            alg.predict(x_test)
        elif hasattr(alg, 'transform'):
            alg.transform(x_test)
        elif hasattr(alg, 'kneighbors'):
            alg.kneighbors(x_test)
    del alg, x_train, x_test, y_train, y_test
    mem_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_diff = mem_after - mem_before

    assert mem_diff < 0.25 * data_memory_size, \
        'Size of extra allocated memory is greater than 25% of input data:' \
        f'\n\tInput data size: {data_memory_size} bytes' \
        f'\n\tExtra allocated memory size: {mem_diff} bytes' \
        f' / {round((mem_diff) / data_memory_size * 100, 2)} %'


@pytest.mark.parametrize('data_transform_function', DATA_TRANSFORMS)
@pytest.mark.parametrize('estimator', ESTIMATORS)
def test_memory_leaks(estimator, data_transform_function):
    _kfold_function_template(estimator, data_transform_function)


if __name__ == '__main__':
    test_memory_leaks()
