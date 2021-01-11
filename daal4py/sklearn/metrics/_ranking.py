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

import daal4py as d4p
import numpy as np
from functools import partial
from collections.abc import Sequence
from scipy.sparse.base import spmatrix

from sklearn.utils import check_array
from sklearn.utils.multiclass import is_multilabel
from sklearn.preprocessing import label_binarize

from ..utils.validation import _daal_assert_all_finite
from .._utils import get_patch_message, sklearn_check_version
import logging

if sklearn_check_version('0.22'):
    from sklearn.metrics._ranking import _multiclass_roc_auc_score as multiclass_roc_auc_score
    from sklearn.metrics._ranking import _binary_roc_auc_score
    from sklearn.metrics._base import _average_binary_score
else:
    from sklearn.metrics.ranking import roc_auc_score as multiclass_roc_auc_score

try:
    import pandas as pd
    pandas_is_imported = True
except ImportError:
    pandas_is_imported = False

def _daal_type_of_target(y):
    valid = ((isinstance(y, (Sequence, spmatrix)) or hasattr(y, '__array__'))
             and not isinstance(y, str))

    if not valid:
        raise ValueError('Expected array-like (array or non-string sequence), '
                         'got %r' % y)

    sparse_pandas = (y.__class__.__name__ in ['SparseSeries', 'SparseArray'])
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if is_multilabel(y):
        return 'multilabel-indicator'

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return 'unknown'

    # The old sequence of sequences format
    try:
        if (not hasattr(y[0], '__array__') and isinstance(y[0], Sequence)
                and not isinstance(y[0], str)):
            raise ValueError('You appear to be using a legacy multi-label data'
                             ' representation. Sequence of sequences are no'
                             ' longer supported; use a binary array or sparse'
                             ' matrix instead - the MultiLabelBinarizer'
                             ' transformer can convert to this format.')
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (y.dtype == object and len(y) != 0 and
                      not isinstance(y.flat[0], str)):
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
        _daal_assert_all_finite(y)
        return 'continuous' + suffix

    unique = np.sort(pd.unique(y.ravel())) if pandas_is_imported else np.unique(y)

    if (len(unique) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        result = ('multiclass' + suffix, None)  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        result = ('binary', unique)  # [1, 2] or [["a"], ["b"]]
    return result


def _daal_roc_auc_score(y_true, y_score, *, average="macro", sample_weight=None,
                  max_fpr=None, multi_class="raise", labels=None):
    y_type = _daal_type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    if y_type[0] == "multiclass" or (y_type[0] == "binary" and
                                  y_score.ndim == 2 and
                                  y_score.shape[1] > 2):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.:
            raise ValueError("Partial AUC computation not available in "
                             "multiclass setting, 'max_fpr' must be"
                             " set to `None`, received `max_fpr={0}` "
                             "instead".format(max_fpr))
        if multi_class == 'raise':
            raise ValueError("multi_class must be in ('ovo', 'ovr')")
        logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
        result = multiclass_roc_auc_score(y_true, y_score, labels,
                                          multi_class, average, sample_weight)
    elif y_type[0] == "binary":
        labels = y_type[1]
        daal_use = max_fpr is None and sample_weight is None and len(labels) == 2
        if daal_use:
            logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("daal"))
            if not np.array_equal(labels, [0, 1]):
                y_true = label_binarize(y_true, classes=labels)[:, 0]
            result = d4p.daal_roc_auc_score(y_true.reshape(-1, 1), y_score.reshape(-1, 1))

        if not daal_use or result == -1:
            y_true = label_binarize(y_true, classes=labels)[:, 0]
            logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
            if sklearn_check_version('0.22'):
                result = _average_binary_score(partial(_binary_roc_auc_score,
                                                       max_fpr=max_fpr),
                                               y_true, y_score, average,
                                               sample_weight=sample_weight)
            else:
                result = multiclass_roc_auc_score(y_true, y_score, average,
                                                  sample_weight=sample_weight,
                                                  max_fpr=max_fpr)
    else:
        logging.info("sklearn.metrics.roc_auc_score: " + get_patch_message("sklearn"))
        if sklearn_check_version('0.22'):
            result = _average_binary_score(partial(_binary_roc_auc_score,
                                                   max_fpr=max_fpr),
                                           y_true, y_score, average,
                                           sample_weight=sample_weight)
        else:
            result = multiclass_roc_auc_score(y_true, y_score, average,
                                              sample_weight=sample_weight,
                                              max_fpr=max_fpr)
    return result
