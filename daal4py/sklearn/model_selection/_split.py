#!/usr/bin/env python
#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
import daal4py as d4p
import numpy as np
from daal4py.sklearn._utils import daal_check_version
import platform

try:
    from sklearn.utils import _safe_indexing as safe_indexing
except ImportError:
    from sklearn.utils import safe_indexing

try:
    import mkl_random
    mkl_random_is_imported = True
except (ImportError, ModuleNotFoundError):
    mkl_random_is_imported = False

try:
    import pandas as pd
    pandas_is_imported = True
except ImportError:
    pandas_is_imported = False


def get_dtypes(data):
    if hasattr(data, 'dtype'):
        return [data.dtype]
    elif hasattr(data, 'dtypes'):
        return list(data.dtypes)
    elif hasattr(data, 'values'):
        return [data.values.dtype]
    return None


def _daal_train_test_split(*arrays, **options):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    random_state = options.pop('random_state', None)
    stratify = options.pop('stratify', None)
    shuffle = options.pop('shuffle', True)
    rng = options.pop('rng', 'OPTIMIZED_MT19937')

    available_rngs = ['default', 'MT19937', 'SFMT19937', 'MT2203', 'R250',
                      'WH', 'MCG31', 'MCG59', 'MRG32K3A', 'PHILOX4X32X10',
                      'NONDETERM', 'OPTIMIZED_MT19937']
    if rng not in available_rngs:
        raise ValueError(
            "Wrong random numbers generator is chosen. "
            "Available generators: %s" % str(available_rngs)[1:-1])

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size,
                                              default_test_size=0.25)
    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for "
                "shuffle=False")

        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
    else:
        if stratify is not None:
            cv = StratifiedShuffleSplit(test_size=n_test,
                                        train_size=n_train,
                                        random_state=random_state)
            train, test = next(cv.split(X=arrays[0], y=stratify))
        else:
            if mkl_random_is_imported and rng not in ['default', 'OPTIMIZED_MT19937'] and (isinstance(random_state, int) or random_state is None):
                random_state = mkl_random.RandomState(random_state, rng)
                indexes = random_state.permutation(n_train + n_test)
                test, train = indexes[:n_test], indexes[n_test:]
            elif rng == 'OPTIMIZED_MT19937' and daal_check_version(((2020,'P', 3), (2021,'B',9))) \
            and (isinstance(random_state, int) or random_state is None) \
            and platform.system() != 'Windows':
                indexes = np.empty(shape=(n_train + n_test,), dtype=np.int64 if n_train + n_test > 2 ** 31 - 1 else np.int32)
                random_state = np.random.RandomState(random_state)
                random_state = random_state.get_state()[1]
                d4p.daal_generate_shuffled_indices([indexes], [random_state])
                test, train = indexes[:n_test], indexes[n_test:]
            else:
                cv = ShuffleSplit(test_size=n_test,
                                  train_size=n_train,
                                  random_state=random_state)
                train, test = next(cv.split(X=arrays[0], y=stratify))

    res = []
    for arr in arrays:
        fallback = False


        # input format check
        if not isinstance(arr, np.ndarray):
            if pandas_is_imported:
                if not isinstance(arr, pd.core.frame.DataFrame) and not isinstance(arr, pd.core.series.Series):
                    fallback = True
            else:
                fallback = True

        # dimensions check
        if hasattr(arr, 'ndim'):
            if arr.ndim > 2:
                fallback = True
        else:
            fallback = True

        # data types check
        dtypes = get_dtypes(arr)
        if dtypes is None:
            fallback = True
        else:
            for i, dtype in enumerate(dtypes):
                if 'float' not in str(dtype) and 'int' not in str(dtype):
                    fallback = True
                    break

        if fallback:
            res.append(safe_indexing(arr, train))
            res.append(safe_indexing(arr, test))
        else:

            if len(arr.shape) == 2:
                n_cols = arr.shape[1]
                reshape_later = False
            else:
                n_cols = 1
                reshape_later = True

            arr_copy = d4p.get_data(arr)
            if not isinstance(arr_copy, list):
                arr_copy = arr_copy.reshape((arr_copy.shape[0], n_cols), order='A')
            if isinstance(arr_copy, np.ndarray):
                order = 'C' if arr_copy.flags['C_CONTIGUOUS'] else 'F'
                train_arr = np.empty(shape=(n_train, n_cols), dtype=arr_copy.dtype, order=order)
                test_arr = np.empty(shape=(n_test, n_cols), dtype=arr_copy.dtype, order=order)
                d4p.daal_train_test_split(arr_copy, train_arr, test_arr, [train], [test])
                if reshape_later:
                    train_arr, test_arr = train_arr.reshape((n_train,)), test_arr.reshape((n_test,))
            elif isinstance(arr_copy, list):
                train_arr = [np.empty(shape=(n_train,), dtype=el.dtype, order='C' if el.flags['C_CONTIGUOUS'] else 'F') for el in arr_copy]
                test_arr = [np.empty(shape=(n_test,), dtype=el.dtype, order='C' if el.flags['C_CONTIGUOUS'] else 'F') for el in arr_copy]
                d4p.daal_train_test_split(arr_copy, train_arr, test_arr, [train], [test])
                train_arr = {col: train_arr[i] for i, col in enumerate(arr.columns)}
                test_arr = {col: test_arr[i] for i, col in enumerate(arr.columns)}
            else:
                raise ValueError('Array can\'t be converted to needed format')

            if pandas_is_imported:
                if isinstance(arr, pd.core.frame.DataFrame):
                    train_arr, test_arr = pd.DataFrame(train_arr), pd.DataFrame(test_arr)
                if isinstance(arr, pd.core.series.Series):
                    train_arr, test_arr = train_arr.reshape(n_train), test_arr.reshape(n_test)
                    train_arr, test_arr = pd.Series(train_arr), pd.Series(test_arr)

            if hasattr(arr, 'index'):
                train_arr.index = train
                test_arr.index = test

            res.append(train_arr)
            res.append(test_arr)

    return res
