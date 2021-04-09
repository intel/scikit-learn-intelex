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
import numbers
import warnings


def _column_or_1d(y):
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
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
    else:
        # user-defined dictionary
        weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'balanced', or None,"
                             " got: %r" % class_weight)
        else:
            for c in class_weight:
                i = np.searchsorted(classes, c)
                if i >= len(classes) or classes[i] != c:
                    raise ValueError("Class label {} not present.".format(c))
                else:
                    weight[i] = class_weight[c]

    return weight


def _validate_targets(y, class_weight, dtype):
    y_ = _column_or_1d(y)
    classes, y = np.unique(y_, return_inverse=True)
    class_weight_res = _compute_class_weight(class_weight,
                                             classes=classes, y=y_)

    if len(classes) < 2:
        raise ValueError(
            "The number of classes has to be greater than one; got %d"
            " class" % len(classes))

    return np.asarray(y, dtype=dtype, order='C'), class_weight_res, classes


def _check_array(array, dtype="numeric", accept_sparse=False, order=None,
                 copy=False, force_all_finite=True, ensure_2d=True):
    # TODO
    from sklearn.utils.validation import check_array
    array = check_array(array=array, dtype=dtype, accept_sparse=accept_sparse,
                        order=order, copy=copy, force_all_finite=force_all_finite,
                        ensure_2d=ensure_2d)

    # TODO: If data is not contiguous copy to contiguous
    # Need implemeted numpy table in oneDAL
    if not array.flags.c_contiguous and not array.flags.f_contiguous:
        array = np.ascontiguousarray(array, array.dtype)
    return array


def _check_X_y(X, y, dtype="numeric", accept_sparse=False, order=None, copy=False,
               force_all_finite=True, ensure_2d=True):
    if y is None:
        raise ValueError("y cannot be None")

    X = _check_array(X, accept_sparse=accept_sparse,
                     dtype=dtype, order=order, copy=copy,
                     force_all_finite=force_all_finite,
                     ensure_2d=ensure_2d)

    y = _column_or_1d(y)
    if y.dtype.kind == 'O':
        y = y.astype(np.float64)
    # TODO: replace on daal4py
    from sklearn.utils.validation import assert_all_finite
    assert_all_finite(y)

    lengths = [len(X), len(y)]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(length) for length in lengths])

    return X, y


def _get_sample_weight(X, y, sample_weight, class_weight, classes):

    n_samples = X.shape[0]
    dtype = X.dtype
    if n_samples == 1:
        raise ValueError("n_samples=1")

    sample_weight = np.asarray([]
                               if sample_weight is None
                               else sample_weight, dtype=np.float64)

    sample_weight_count = sample_weight.shape[0]
    if sample_weight_count != 0 and sample_weight_count != n_samples:
        raise ValueError("sample_weight and X have incompatible shapes: "
                         "%r vs %r\n"
                         "Note: Sparse matrices cannot be indexed w/"
                         "boolean masks (use `indices=True` in CV)."
                         % (len(sample_weight), X.shape))

    ww = None
    if sample_weight_count == 0 and class_weight is None:
        return ww
    elif sample_weight_count == 0:
        sample_weight = np.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
    else:
        sample_weight = _check_array(
            sample_weight, accept_sparse=False, ensure_2d=False,
            dtype=dtype, order="C"
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError("sample_weight.shape == {}, expected {}!"
                             .format(sample_weight.shape, (n_samples,)))
    if np.all(sample_weight <= 0):
        raise ValueError(
            'Invalid input - all samples have zero or negative weights.')
    elif np.any(sample_weight <= 0):
        if len(np.unique(y[sample_weight > 0])) != len(classes):
            raise ValueError(
                'Invalid input - all samples with positive weights '
                'have the same label.')
    ww = sample_weight
    if class_weight is not None:
        for i, v in enumerate(class_weight):
            ww[y == i] *= v

    if not ww.flags.c_contiguous and not ww.flags.f_contiguous:
        ww = np.ascontiguousarray(ww, dtype)

    return ww


def _check_is_fitted(estimator, attributes=None, *, msg=None):
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise AttributeError(msg % {'name': type(estimator).__name__})
