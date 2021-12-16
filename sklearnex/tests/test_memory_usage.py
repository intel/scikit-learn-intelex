#===============================================================================
# Copyright 2021-2022 Intel Corporation
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
import types
import tracemalloc
from sklearnex import get_patch_map
from sklearnex.model_selection import train_test_split
from sklearnex.utils import assert_all_finite
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification, make_regression
import pandas as pd
import numpy as np
import gc
import logging


class TrainTestSplitEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        train_test_split(x, y)


class FiniteCheckEstimator:
    def __init__(self):
        pass

    def fit(self, x, y):
        assert_all_finite(x)
        assert_all_finite(y)


ESTIMATORS = [
    TrainTestSplitEstimator,
    FiniteCheckEstimator
]


BANNED_ESTIMATORS = [
    'TSNE',  # too slow for using in testing on applicable data sizes
    'RandomForestClassifier',  # Failed, need to investigate and fix this issue
    'RandomForestRegressor',  # Failed, need to investigate and fix this issue
]

# add all daa4lpy estimators enabled in patching (except banned)
patched_estimators = get_patch_map().values()
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
    return pd.DataFrame(np.ascontiguousarray(x)), pd.Series(y)


def dataframe_f(x, y):
    return pd.DataFrame(np.asfortranarray(x)), pd.Series(y)


DATA_TRANSFORMS = [
    ndarray_c,
    ndarray_f,
    dataframe_c,
    dataframe_f
]
EXTRA_MEMORY_THRESHOLD = 0.20


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
    mem_before_gc, _ = tracemalloc.get_traced_memory()
    mem_diff = mem_before_gc - mem_before
    message = 'Size of extra allocated memory {} using garbage collector' \
        f'is greater than {EXTRA_MEMORY_THRESHOLD * 100}% of input data:' \
        f'\n\tAlgorithm: {estimator.__name__}' \
        f'\n\tInput data size: {data_memory_size} bytes' \
        f'\n\tExtra allocated memory size: {mem_diff} bytes' \
        ' / {} %'
    if mem_diff >= EXTRA_MEMORY_THRESHOLD * data_memory_size:
        logging.warning(message.format(
            'before', round((mem_diff) / data_memory_size * 100, 2)))
    gc.collect()
    mem_after, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_diff = mem_after - mem_before

    assert mem_diff < EXTRA_MEMORY_THRESHOLD * data_memory_size, \
        message.format('after', round((mem_diff) / data_memory_size * 100, 2))


@pytest.mark.parametrize('data_transform_function', DATA_TRANSFORMS)
@pytest.mark.parametrize('estimator', ESTIMATORS)
def test_memory_leaks(estimator, data_transform_function):
    _kfold_function_template(estimator, data_transform_function)


if __name__ == '__main__':
    test_memory_leaks()
