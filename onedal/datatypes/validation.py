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

import numpy as np
import warnings
from scipy import sparse as sp
from scipy.sparse import issparse, dok_matrix, lil_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array
from collections.abc import Sequence
from numbers import Integral
from daal4py.sklearn.utils.validation import _daal_assert_all_finite as assert_all_finite


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.
    """


def _column_or_1d(y, warn=False):
    y = np.asarray(y)

    # TODO: Convert this kind of arrays to a table like in daal4py
    if not y.flags.aligned and not y.flags.writeable:
        y = np.array(y.tolist())

    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn("A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().",
                          DataConversionWarning, stacklevel=2)
        return np.ravel(y)

    raise ValueError(
        "y should be a 1d array, "
        "got an array of shape {} instead.".format(shape))


def _compute_class_weight(class_weight, classes, y):
    if set(y) - set(classes):
        raise ValueError("classes should include all valid labels that can "
                         "be in y")
    if class_weight is None or len(class_weight) == 0:
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
    elif class_weight == 'balanced':
        y_ = _column_or_1d(y)
        classes, _ = np.unique(y_, return_inverse=True)

        le = LabelEncoder()
        y_ind = le.fit_transform(y_)
        if not all(np.in1d(classes, le.classes_)):
            raise ValueError("classes should have valid labels that are in y")

        y_bin = np.bincount(y_ind).astype(np.float64)
        weight = len(y_) / (len(le.classes_) * y_bin)
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            i = np.searchsorted(classes, c)
            if i >= len(classes) or classes[i] != c:
                raise ValueError("Class label {} not present.".format(c))
            weight[i] = class_weight[c]

    return weight


def _validate_targets(y, class_weight, dtype):
    y_ = _column_or_1d(y, warn=True)
    _check_classification_targets(y)
    classes, y = np.unique(y_, return_inverse=True)
    class_weight_res = _compute_class_weight(class_weight,
                                             classes=classes, y=y_)

    if len(classes) < 2:
        raise ValueError(
            "The number of classes has to be greater than one; got %d"
            " class" % len(classes))

    return np.asarray(y, dtype=dtype, order='C'), class_weight_res, classes


def _check_array(array, dtype="numeric", accept_sparse=False, order=None,
                 copy=False, force_all_finite=True,
                 ensure_2d=True, accept_large_sparse=True):
    if force_all_finite:
        if sp.issparse(array):
            if hasattr(array, 'data'):
                assert_all_finite(array.data)
                force_all_finite = False
        else:
            assert_all_finite(array)
            force_all_finite = False
    array = check_array(
        array=array,
        dtype=dtype,
        accept_sparse=accept_sparse,
        order=order,
        copy=copy,
        force_all_finite=force_all_finite,
        ensure_2d=ensure_2d,
        accept_large_sparse=accept_large_sparse)

    if sp.isspmatrix(array):
        return array

    # TODO: Convert this kind of arrays to a table like in daal4py
    if not array.flags.aligned and not array.flags.writeable:
        array = np.array(array.tolist())

    # TODO: If data is not contiguous copy to contiguous
    # Need implemeted numpy table in oneDAL
    if not array.flags.c_contiguous and not array.flags.f_contiguous:
        array = np.ascontiguousarray(array, array.dtype)
    return array


def _check_X_y(
        X,
        y,
        dtype="numeric",
        accept_sparse=False,
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        accept_large_sparse=True,
        y_numeric=False,
        accept_2d_y=False):
    if y is None:
        raise ValueError("y cannot be None")

    X = _check_array(X, accept_sparse=accept_sparse,
                     dtype=dtype, order=order, copy=copy,
                     force_all_finite=force_all_finite,
                     ensure_2d=ensure_2d,
                     accept_large_sparse=accept_large_sparse)

    if not accept_2d_y:
        y = _column_or_1d(y, warn=True)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)
    assert_all_finite(y)

    lengths = [X.shape[0], y.shape[0]]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(length) for length in lengths])

    return X, y


def _check_classification_targets(y):
    y_type = _type_of_target(y)
    if y_type not in ['binary', 'multiclass', 'multiclass-multioutput',
                      'multilabel-indicator', 'multilabel-sequences']:
        raise ValueError("Unknown label type: %r" % y_type)


def _type_of_target(y):
    is_sequence, is_array = isinstance(y, Sequence), hasattr(y, '__array__')
    is_not_string, is_spmatrix = not isinstance(y, str), sp.isspmatrix(y)
    valid = (is_sequence or is_array or is_spmatrix) and is_not_string

    if not valid:
        raise ValueError('Expected array-like (array or non-string sequence), '
                         'got %r' % y)

    sparse_pandas = (y.__class__.__name__ in ['SparseSeries', 'SparseArray'])
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if _is_multilabel(y):
        return 'multilabel-indicator'

    # DeprecationWarning will be replaced by ValueError, see NEP 34
    # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
    with warnings.catch_warnings():
        warnings.simplefilter('error', np.VisibleDeprecationWarning)
        try:
            y = np.asarray(y)
        except np.VisibleDeprecationWarning:
            # dtype=object should be provided explicitly for ragged arrays,
            # see NEP 34
            y = np.asarray(y, dtype=object)

    # The old sequence of sequences format
    try:
        if not hasattr(y[0], '__array__') and isinstance(y[0], Sequence) \
                and not isinstance(y[0], str):
            raise ValueError('You appear to be using a legacy multi-label data'
                             ' representation. Sequence of sequences are no'
                             ' longer supported; use a binary array or sparse'
                             ' matrix instead - the MultiLabelBinarizer'
                             ' transformer can convert to this format.')
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (y.dtype == object and len(
            y) and not isinstance(y.flat[0], str)):
        return 'unknown'  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return 'unknown'  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # check float and contains non-integer float values
    if y.dtype.kind == 'f' and np.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        assert_all_finite(y)
        return 'continuous' + suffix

    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass' + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    return 'binary'  # [1, 2] or [["a"], ["b"]]


def _is_integral_float(y):
    return y.dtype.kind == 'f' and np.all(y.astype(int) == y)


def _is_multilabel(y):
    if hasattr(y, '__array__') or isinstance(y, Sequence):
        # DeprecationWarning will be replaced by ValueError, see NEP 34
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.VisibleDeprecationWarning)
            try:
                y = np.asarray(y)
            except np.VisibleDeprecationWarning:
                # dtype=object should be provided explicitly for ragged arrays,
                # see NEP 34
                y = np.array(y, dtype=object)

    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
        return len(y.data) == 0 or np.unique(y.data).size == 1 and \
            (y.dtype.kind in 'biu' or _is_integral_float(np.unique(y.data)))
    labels = np.unique(y)

    return len(labels) < 3 and (
        y.dtype.kind in 'biu' or _is_integral_float(labels))


def _check_n_features(self, X, reset):
    try:
        n_features = _num_features(X)
    except TypeError as e:
        if not reset and hasattr(self, "n_features_in_"):
            raise ValueError(
                "X does not contain any features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features"
            ) from e
        # If the number of features is not defined and reset=True,
        # then we skip this check
        return

    if reset:
        self.n_features_in_ = n_features
        return

    if not hasattr(self, "n_features_in_"):
        # Skip this check if the expected number of expected input features
        # was not recorded by calling fit first. This is typically the case
        # for stateless transformers.
        return

    if n_features != self.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {self.__class__.__name__} "
            f"is expecting {self.n_features_in_} features as input.")


def _num_features(X, fallback_1d=False):
    type_ = type(X)
    if type_.__module__ == "builtins":
        type_name = type_.__qualname__
    else:
        type_name = f"{type_.__module__}.{type_.__qualname__}"
    message = (
        "Unable to find the number of features from X of type "
        f"{type_name}"
    )
    if not hasattr(X, '__len__') and not hasattr(X, 'shape'):
        if not hasattr(X, '__array__'):
            raise TypeError(message)
        # Only convert X to a numpy array if there is no cheaper, heuristic
        # option.
        X = np.asarray(X)

    if hasattr(X, 'shape'):
        ndim_thr = 1 if fallback_1d else 2
        if not hasattr(X.shape, '__len__') or len(X.shape) < ndim_thr:
            message += f" with shape {X.shape}"
            raise TypeError(message)
        return X.shape[-1]

    first_sample = X[0]

    # Do not consider an array-like of strings or dicts to be a 2D array
    if isinstance(first_sample, (str, bytes, dict)):
        message += (f" where the samples are of type "
                    f"{type(first_sample).__qualname__}")
        raise TypeError(message)

    try:
        # If X is a list of lists, for instance, we assume that all nested
        # lists have the same length without checking or converting to
        # a numpy array to keep this function call as cheap as possible.
        if (not fallback_1d) or hasattr(first_sample, '__len__'):
            return len(first_sample)
        else:
            return 1
    except Exception as err:
        raise TypeError(message) from err


def _num_samples(x):
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." %
                x)
    # Check that shape is returning an integer or default to len
    # Dask dataframes may not return numeric shape[0] value
    if hasattr(x, "shape") and isinstance(x.shape[0], Integral):
        return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


def _get_2d_shape(x, fallback_1d=True):
    n_samples = _num_samples(x)
    n_features = _num_features(x, fallback_1d)
    return (n_samples, n_features)
