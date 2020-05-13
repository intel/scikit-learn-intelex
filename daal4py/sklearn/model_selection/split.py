from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
import daal4py as d4p
import numpy as np

from timeit import default_timer as timer

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
    else:
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
    rng = options.pop('rng', 'MT19937')

    available_rngs = ['MT19937', 'SFMT19937', 'MT2203', 'R250', 'WH', 'MCG31',
                      'MCG59', 'MRG32K3A', 'PHILOX4X32X10', 'NONDETERM']
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
            if mkl_random_is_imported and (isinstance(random_state, int) or random_state is None):
                random_state = mkl_random.RandomState(random_state, rng)
                indexes = random_state.permutation(n_train + n_test)
                train, test = indexes[:n_train], indexes[n_train:]
            else:
                cv = ShuffleSplit(test_size=n_test,
                                  train_size=n_train,
                                  random_state=random_state)
                train, test = next(cv.split(X=arrays[0], y=stratify))

    res = []
    for arr in arrays:
        fallback = False
        if isinstance(arr, np.ndarray):
            origin_type = "numpy"
        elif pandas_is_imported:
            if isinstance(arr, pd.core.frame.DataFrame):
                origin_type = "pandas DataFrame"
            elif isinstance(arr, pd.core.series.Series):
                origin_type = "pandas Series"
            else:
                fallback = True
        else:
            fallback = True

        dtypes = get_dtypes(arr)
        if dtypes is None:
            fallback = True
        else:
            for i, dtype in enumerate(dtypes):
                if dtype in [np.int32, 'int32']:
                    dtypes[i] = 0
                elif dtype in [np.float32, 'float32']:
                    dtypes[i] = 1
                elif dtype in [np.float64, 'float64']:
                    dtypes[i] = 2
                else:
                    fallback = True
                    break
        if fallback:
            train = train.reshape((n_train,))
            test = test.reshape((n_test,))
            res.append(_safe_indexing(arr, train))
            res.append(_safe_indexing(arr, test))
        else:
            train = train.reshape(n_train, 1)
            test = test.reshape(n_test, 1)

            if len(arr.shape) == 2:
                n_cols = arr.shape[1]
                reshape_later = False
            else:
                n_cols = 1
                reshape_later = True

            unique_dtypes = set(dtypes)
            dtypes_list = dtypes
            if origin_type == 'pandas DataFrame':
                if len(unique_dtypes) > 1:
                    origin_type += ' with multiple dtypes'
                else:
                    origin_type += ' with one dtype'
            if len(dtypes) == 1:
                dtypes = dtypes * n_cols
            dtypes = np.array(dtypes, dtype=np.int32, ndmin=2)

            if origin_type == 'numpy':
                order = 'C' if arr.flags['C_CONTIGUOUS'] == True else 'F'

                train_arr = np.empty(shape=(n_train, n_cols), dtype=arr.dtype, order=order)
                test_arr = np.empty(shape=(n_test, n_cols), dtype=arr.dtype, order=order)

                d4p.daal_train_test_split(arr.reshape((arr.shape[0], n_cols), order='A'), train_arr, test_arr, train, test, dtypes)

                if reshape_later:
                    train_arr = train_arr.reshape((train_arr.shape[0],))
                    test_arr = test_arr.reshape((test_arr.shape[0],))
                    arr = arr.reshape((arr.shape[0],), order='A')

            elif origin_type == 'pandas DataFrame with one dtype':
                order = 'C' if arr.values.flags['C_CONTIGUOUS'] == True else 'F'

                train_arr = np.empty(shape=(n_train, n_cols), dtype=arr.values.dtype, order=order)
                test_arr = np.empty(shape=(n_test, n_cols), dtype=arr.values.dtype, order=order)

                d4p.daal_train_test_split(arr.values, train_arr, test_arr, train, test, dtypes)

                train_arr, test_arr = pd.DataFrame(train_arr), pd.DataFrame(test_arr)

            elif origin_type == 'pandas DataFrame with multiple dtypes':
                train_arr, test_arr = {}, {}
                for i, col in enumerate(list(arr.columns)):
                    train_arr[col] = np.empty(shape=(n_train,), dtype=arr.dtypes[col])
                    test_arr[col] = np.empty(shape=(n_test,), dtype=arr.dtypes[col])
                train_arr = pd.DataFrame(train_arr)
                test_arr = pd.DataFrame(test_arr)
                for i, col in enumerate(list(arr.columns)):
                    d4p.daal_train_test_split(
                        arr[col].values.reshape(n_train + n_test, 1),
                        train_arr[col].values.reshape(n_train, 1),
                        test_arr[col].values.reshape(n_test, 1),
                        train, test, np.array([dtypes_list[i]], dtype=np.int32, ndmin=2))

            else:
                train_arr = np.empty(shape=(n_train, 1), dtype=arr.dtype)
                test_arr = np.empty(shape=(n_test, 1), dtype=arr.dtype)

                orig = arr.array.to_numpy().reshape((arr.shape[0], 1), order='A')

                d4p.daal_train_test_split(orig, train_arr, test_arr, train, test, dtypes)

                train_arr = train_arr.reshape((n_train,))
                test_arr = test_arr.reshape((n_test,))

                train_arr, test_arr = pd.Series(train_arr), pd.Series(test_arr)

            if hasattr(arr, 'index'):
                train = train.reshape((n_train,))
                test = test.reshape((n_test,))
                train_arr.index = train
                test_arr.index = test

            res.append(train_arr)
            res.append(test_arr)

    return res
