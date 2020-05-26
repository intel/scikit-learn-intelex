from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn.model_selection._split import _validate_shuffle_split
import daal4py as d4p
import numpy as np

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
        if not isinstance(arr, np.ndarray):
            if pandas_is_imported:
                if not isinstance(arr, pd.core.frame.DataFrame) and not isinstance(arr, pd.core.series.Series):
                    fallback = True
            else:
                fallback = True

        dtypes = get_dtypes(arr)
        if dtypes is None:
            fallback = True
        else:
            for i, dtype in enumerate(dtypes):
                if dtype not in [np.int32, np.float32, np.float64, 'int32', 'float32', 'float64']:
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
                one_dim = False
                n_cols = arr.shape[1]
            else:
                one_dim = True
                n_cols = 1

            arr_copy = d4p.get_data(arr)
            if not isinstance(arr_copy, list):
                arr_copy = arr_copy.reshape((arr_copy.shape[0], n_cols), order='A')
            if isinstance(arr_copy, np.ndarray):
                order = 'C' if arr_copy.flags['C_CONTIGUOUS'] == True else 'F'
                train_arr = np.empty(shape=(n_train, n_cols), dtype=arr_copy.dtype, order=order)
                test_arr = np.empty(shape=(n_test, n_cols), dtype=arr_copy.dtype, order=order)
                d4p.daal_train_test_split(arr_copy, train_arr, test_arr, train, test)
            elif isinstance(arr_copy, list):
                train_arr = [np.empty(shape=(n_train,), dtype=el.dtype, order='C' if el.flags['C_CONTIGUOUS'] else 'F') for el in arr_copy]
                test_arr = [np.empty(shape=(n_test,), dtype=el.dtype, order='C' if el.flags['C_CONTIGUOUS'] else 'F') for el in arr_copy]
                d4p.daal_train_test_split(arr_copy, train_arr, test_arr, train, test)
                train_arr = {col: train_arr[i] for i, col in enumerate(arr.columns)}
                test_arr = {col: test_arr[i] for i, col in enumerate(arr.columns)}
            else:
                raise ValueError('Array can\'t be converted to needed format')

            if isinstance(arr, pd.core.frame.DataFrame):
                train_arr, test_arr = pd.DataFrame(train_arr), pd.DataFrame(test_arr)
            if isinstance(arr, pd.core.series.Series):
                train_arr, test_arr = train_arr.reshape(n_train), test_arr.reshape(n_test)
                train_arr, test_arr = pd.Series(train_arr), pd.Series(test_arr)

            if hasattr(arr, 'index'):
                train = train.reshape((n_train,))
                test = test.reshape((n_test,))
                train_arr.index = train
                test_arr.index = test

            res.append(train_arr)
            res.append(test_arr)

    return res
