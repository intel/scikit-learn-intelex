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

import numpy as np

from daal4py import _get__daal_link_version__ as dv
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
import daal4py

def daal_check_version(rule):
    # First item is major version - 2021, second is minor+patch - 0110, third item is status - B
    target = (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))
    if not isinstance(rule[0], type(target)):
        if rule > target:
            return False
    else:
        for rule_item in rule:
            if rule_item > target:
                return False
            if rule_item[0]==target[0]:
                break
    return True

def sklearn_check_version(ver):
    return bool(LooseVersion(sklearn_version) >= LooseVersion(ver))

def get_daal_version():
    return (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))

def parse_dtype(dt):
    if dt == np.double:
        return "double"
    elif dt == np.single:
        return "float"
    raise ValueError(f"Input array has unexpected dtype = {dt}")

def getFPType(X):
    try:
        from pandas import DataFrame
        from pandas.core.dtypes.cast import find_common_type
        if isinstance(X, DataFrame):
            dt = find_common_type(X.dtypes)
            return parse_dtype(dt)
    except ImportError:
        pass

    dt = getattr(X, 'dtype', None)
    return parse_dtype(dt)

def make2d(X):
    if np.isscalar(X):
        X = np.asarray(X)[np.newaxis, np.newaxis]
    elif isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape((X.size, 1))
    return X

def get_patch_message(s):
    if s == "daal":
        message = "uses Intel(R) oneAPI Data Analytics Library solver"
    elif s == "sklearn":
        message = "uses original Scikit-learn solver"
    elif s == "sklearn_after_daal":
        message = "uses original Scikit-learn solver, because the task was not solved with Intel(R) oneAPI Data Analytics Library"
    else:
        raise ValueError(f"Invalid input - expected one of 'daal','sklearn', 'sklearn_after_daal', got {s}")
    return message

def is_in_sycl_ctxt():
    try:
        from daal4py.oneapi import is_in_sycl_ctxt as is_in_ctx
        return is_in_ctx()
    except ModuleNotFoundError:
        return False

def is_DataFrame(X):
    try:
        from pandas import DataFrame
        return isinstance(X, DataFrame)
    except ImportError:
        return False

def get_dtype(X):
    try:
        from pandas.core.dtypes.cast import find_common_type
        return find_common_type(X.dtypes) if is_DataFrame(X) else X.dtype
    except ImportError:
        return getattr(X, "dtype", None)

def get_number_of_types(dataframe):
    dtypes = getattr(dataframe, "dtypes", None)
    return 1 if dtypes is None else len(set(dtypes))
