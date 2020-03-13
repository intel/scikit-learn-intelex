#
#*******************************************************************************
# Copyright 2014-2020 Intel Corporation
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
#******************************************************************************/

import numpy as np
import daal4py as d4p
from sklearn import get_config as _get_config
from sklearn.utils.fixes import _object_dtype_isnan

def _daal_assert_all_finite(X, allow_nan=False, msg_dtype=None):
    """Like assert_all_finite, but only for ndarray."""
    # validation is also imported in extmath
    from sklearn.utils.extmath import _safe_accumulator_op

    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)

    dt = X.dtype
    is_float = dt.kind in 'fc'

    msg_err = "Input contains {} or a value too large for {!r}."
    type_err = 'infinity' if allow_nan else 'NaN, infinity'
    err = msg_err.format(type_err, msg_dtype if msg_dtype is not None else X.dtype)

    if (X.ndim in [1, 2]
        and not np.any(np.equal(X.shape, 0))
        and dt in [np.float32, np.float64]
        ):
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if dt == np.float64:
            if not d4p.daal_assert_all_finite(X, allow_nan, 0):
                raise ValueError(err)
        elif dt == np.float32:
            if not d4p.daal_assert_all_finite(X, allow_nan, 1):
                raise ValueError(err)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    elif is_float and (np.isfinite(_safe_accumulator_op(np.sum, X))):
        pass
    elif is_float:
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            raise ValueError(err)
    # for object dtype data, we only check for NaNs (GH-13254)
    elif X.dtype == np.dtype('object') and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")
