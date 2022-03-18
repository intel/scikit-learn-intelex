#===============================================================================
# Copyright 2014 Intel Corporation
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
import sys

from daal4py import _get__daal_link_version__ as dv
from sklearn import __version__ as sklearn_version
try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version
import logging


def set_idp_sklearn_verbose():
    import warnings
    import os
    logLevel = os.environ.get("IDP_SKLEARN_VERBOSE")
    try:
        if logLevel is not None:
            logging.basicConfig(
                stream=sys.stdout,
                format='%(levelname)s: %(message)s', level=logLevel.upper())
    except Exception:
        warnings.warn('Unknown level "{}" for logging.\n'
                      'Please, use one of "CRITICAL", "ERROR", '
                      '"WARNING", "INFO", "DEBUG".'.format(logLevel))


def daal_check_version(rule):
    # First item is major version - 2021,
    # second is minor+patch - 0110,
    # third item is status - B
    target = (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))
    if not isinstance(rule[0], type(target)):
        if rule > target:
            return False
    else:
        for rule_item in rule:
            if rule_item > target:
                return False
            if rule_item[0] == target[0]:
                break
    return True


def sklearn_check_version(ver):
    if hasattr(Version(ver), 'base_version'):
        base_sklearn_version = Version(sklearn_version).base_version
        return bool(Version(base_sklearn_version) >= Version(ver))
    # packaging module not available
    return bool(Version(sklearn_version) >= Version(ver))


def get_daal_version():
    return (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))


def parse_dtype(dt):
    if dt == np.double:
        return "double"
    if dt == np.single:
        return "float"
    raise ValueError(f"Input array has unexpected dtype = {dt}")


def getFPType(X):
    try:
        from pandas import DataFrame
        from pandas.core.dtypes.cast import find_common_type
        if isinstance(X, DataFrame):
            dt = find_common_type(X.dtypes.tolist())
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
        message = "running accelerated version on "
        if 'daal4py.oneapi' in sys.modules:
            from daal4py.oneapi import _get_device_name_sycl_ctxt
            dev = _get_device_name_sycl_ctxt()
            if dev == 'cpu' or dev == 'host' or dev is None:
                message += 'CPU'
            elif dev == 'gpu':
                message += 'GPU'
            else:
                raise ValueError(f"Unexpected device name {dev}."
                                 " Supported types are host, cpu and gpu")
        else:
            message += 'CPU'

    elif s == "sklearn":
        message = "fallback to original Scikit-learn"
    elif s == "sklearn_after_daal":
        message = "failed to run accelerated version, fallback to original Scikit-learn"
    else:
        raise ValueError(
            f"Invalid input - expected one of 'daal','sklearn',"
            f" 'sklearn_after_daal', got {s}")
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
        return find_common_type(list(X.dtypes)) if is_DataFrame(X) else X.dtype
    except ImportError:
        return getattr(X, "dtype", None)


def get_number_of_types(dataframe):
    dtypes = getattr(dataframe, "dtypes", None)
    try:
        return len(set(dtypes))
    except TypeError:
        return 1


class PatchingConditionsChain:
    def __init__(self, scope_name):
        self.scope_name = scope_name
        self.patching_is_enabled = True
        self.messages = []

    def _iter_conditions(self, conditions_and_messages):
        result = []
        for condition, message in conditions_and_messages:
            result.append(condition)
            if not condition:
                self.messages.append(message)
        return result

    def and_conditions(self, conditions_and_messages, conditions_merging=all):
        self.patching_is_enabled &= conditions_merging(
            self._iter_conditions(conditions_and_messages))
        return self.patching_is_enabled

    def or_conditions(self, conditions_and_messages, conditions_merging=all):
        self.patching_is_enabled |= conditions_merging(
            self._iter_conditions(conditions_and_messages))
        return self.patching_is_enabled

    def get_status(self):
        return self.patching_is_enabled

    def write_log(self):
        if self.patching_is_enabled:
            logging.info(f"{self.scope_name}: {get_patch_message('daal')}")
        else:
            logging.debug(
                f'{self.scope_name}: debugging for the patch is enabled to track'
                ' the usage of Intel® oneAPI Data Analytics Library (oneDAL)')
            for message in self.messages:
                logging.debug(
                    f'{self.scope_name}: patching failed with cause - {message}')
            logging.info(f"{self.scope_name}: {get_patch_message('sklearn')}")
