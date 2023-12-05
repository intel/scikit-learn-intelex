# ==============================================================================
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
# ==============================================================================

import os
import sys
import threading
import warnings
from functools import wraps
from inspect import Parameter, signature
from multiprocessing import cpu_count
from numbers import Integral
from typing import Any, Callable, Tuple
from warnings import warn

import numpy as np
import threadpoolctl
from numpy.lib.recfunctions import require_fields
from sklearn import __version__ as sklearn_version

from daal4py import _get__daal_link_version__ as dv
from daal4py import daalinit as set_n_threads
from daal4py import num_threads as get_n_threads

DaalVersionTuple = Tuple[int, str, int]

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

import logging

try:
    from pandas import DataFrame
    from pandas.core.dtypes.cast import find_common_type

    pandas_is_imported = True
except (ImportError, ModuleNotFoundError):
    pandas_is_imported = False

try:
    from daal4py.oneapi import is_in_sycl_ctxt as is_in_ctx

    ctx_imported = True
except (ImportError, ModuleNotFoundError):
    ctx_imported = False

oneapi_is_available = "daal4py.oneapi" in sys.modules
if oneapi_is_available:
    from daal4py.oneapi import _get_device_name_sycl_ctxt


def set_idp_sklearn_verbose():
    logLevel = os.environ.get("IDP_SKLEARN_VERBOSE")
    try:
        if logLevel is not None:
            logging.basicConfig(
                stream=sys.stdout,
                format="%(levelname)s: %(message)s",
                level=logLevel.upper(),
            )
    except Exception:
        warnings.warn(
            'Unknown level "{}" for logging.\n'
            'Please, use one of "CRITICAL", "ERROR", '
            '"WARNING", "INFO", "DEBUG".'.format(logLevel)
        )


def get_daal_version() -> DaalVersionTuple:
    return int(dv()[0:4]), str(dv()[10:11]), int(dv()[4:8])


def daal_check_version(
    required_version: Tuple[Any, ...],
    _get_daal_version: Callable[[], DaalVersionTuple] = get_daal_version,
) -> bool:
    """Check daal version provided as (MAJOR, STATUS, MINOR+PATCH)

    This function also accepts a list or tuple of daal versions. It will return true if
    any version in the list/tuple is <= `_get_daal_version()`.
    """
    if isinstance(required_version[0], (list, tuple)):
        # a list of version candidates was provided, recursively check if any is <= _get_daal_version
        return any(
            map(lambda ver: daal_check_version(ver, _get_daal_version), required_version)
        )

    major_required, status_required, patch_required = required_version
    major, status, patch = _get_daal_version()

    if status != status_required:
        return False

    if major_required < major:
        return True
    if major == major_required:
        return patch_required <= patch

    return False


sklearn_versions_map = {}


def sklearn_check_version(ver):
    if ver in sklearn_versions_map.keys():
        return sklearn_versions_map[ver]
    if hasattr(Version(ver), "base_version"):
        base_sklearn_version = Version(sklearn_version).base_version
        res = bool(Version(base_sklearn_version) >= Version(ver))
    else:
        # packaging module not available
        res = bool(Version(sklearn_version) >= Version(ver))
    sklearn_versions_map[ver] = res
    return res


def parse_dtype(dt):
    if dt == np.double:
        return "double"
    if dt == np.single:
        return "float"
    raise ValueError(f"Input array has unexpected dtype = {dt}")


def getFPType(X):
    if pandas_is_imported:
        if isinstance(X, DataFrame):
            dt = find_common_type(X.dtypes.tolist())
            return parse_dtype(dt)

    dt = getattr(X, "dtype", None)
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
        if oneapi_is_available:
            dev = _get_device_name_sycl_ctxt()
            if dev == "cpu" or dev is None:
                message += "CPU"
            elif dev == "gpu":
                message += "GPU"
            else:
                raise ValueError(
                    f"Unexpected device name {dev}." " Supported types are cpu and gpu"
                )
        else:
            message += "CPU"

    elif s == "sklearn":
        message = "fallback to original Scikit-learn"
    elif s == "sklearn_after_daal":
        message = "failed to run accelerated version, fallback to original Scikit-learn"
    else:
        raise ValueError(
            f"Invalid input - expected one of 'daal','sklearn',"
            f" 'sklearn_after_daal', got {s}"
        )
    return message


def is_in_sycl_ctxt():
    if ctx_imported:
        return is_in_ctx()
    else:
        return False


def is_DataFrame(X):
    if pandas_is_imported:
        return isinstance(X, DataFrame)
    else:
        return False


def get_dtype(X):
    if pandas_is_imported:
        return find_common_type(list(X.dtypes)) if is_DataFrame(X) else X.dtype
    else:
        return getattr(X, "dtype", None)


def get_number_of_types(dataframe):
    dtypes = getattr(dataframe, "dtypes", None)
    try:
        return len(set(dtypes))
    except TypeError:
        return 1


def check_tree_nodes(tree_nodes):
    def convert_to_old_tree_nodes(tree_nodes):
        # conversion from sklearn>=1.3 tree nodes format to previous format:
        # removal of 'missing_go_to_left' field from node dtype
        new_field = "missing_go_to_left"
        new_dtype = tree_nodes.dtype
        old_dtype = np.dtype(
            [
                (key, value[0])
                for key, value in new_dtype.fields.items()
                if key != new_field
            ]
        )
        return require_fields(tree_nodes, old_dtype)

    if sklearn_check_version("1.3"):
        return tree_nodes
    else:
        return convert_to_old_tree_nodes(tree_nodes)


class PatchingConditionsChain:
    def __init__(self, scope_name):
        self.scope_name = scope_name
        self.patching_is_enabled = True
        self.messages = []
        self.logger = logging.getLogger("sklearnex")

    def _iter_conditions(self, conditions_and_messages):
        result = []
        for condition, message in conditions_and_messages:
            result.append(condition)
            if not condition:
                self.messages.append(message)
        return result

    def and_conditions(self, conditions_and_messages, conditions_merging=all):
        self.patching_is_enabled &= conditions_merging(
            self._iter_conditions(conditions_and_messages)
        )
        return self.patching_is_enabled

    def and_condition(self, condition, message):
        return self.and_conditions([(condition, message)])

    def or_conditions(self, conditions_and_messages, conditions_merging=all):
        self.patching_is_enabled |= conditions_merging(
            self._iter_conditions(conditions_and_messages)
        )
        return self.patching_is_enabled

    def write_log(self):
        if self.patching_is_enabled:
            self.logger.info(f"{self.scope_name}: {get_patch_message('daal')}")
        else:
            self.logger.debug(
                f"{self.scope_name}: debugging for the patch is enabled to track"
                " the usage of Intel® oneAPI Data Analytics Library (oneDAL)"
            )
            for message in self.messages:
                self.logger.debug(
                    f"{self.scope_name}: patching failed with cause - {message}"
                )
            self.logger.info(f"{self.scope_name}: {get_patch_message('sklearn')}")

    def get_status(self, logs=False):
        if logs:
            self.write_log()
        return self.patching_is_enabled


def control_n_jobs(original_class):
    """Decorator for the control of 'n_jobs' parameter in estimator class. It applied
    for all estimators with and without support of parameter in original sklearn.
    In case of estimator without 'n_jobs' support, this decorator adds it.
    """
    original_init = original_class.__init__

    if sklearn_check_version("1.2") and hasattr(original_class, "_parameter_constraints"):
        parameter_constraints = original_class._parameter_constraints
        if "n_jobs" not in parameter_constraints:
            parameter_constraints["n_jobs"] = [Integral, None]

    @wraps(original_init)
    def init_with_n_jobs(self, *args, n_jobs=None, **kwargs):
        original_init(self, *args, **kwargs)
        self.n_jobs = n_jobs

    # add "n_jobs" parameter to signature of wrapped init
    # if estimator doesn't originally support it
    sig = signature(original_init)
    original_params = list(sig.parameters.values())
    if "n_jobs" not in list(map(lambda param: param.name, original_params)):
        original_params.append(Parameter("n_jobs", Parameter.KEYWORD_ONLY, default=None))
        init_with_n_jobs.__signature__ = sig.replace(parameters=original_params)
        original_class.__init__ = init_with_n_jobs

    # add n_jobs to __doc__ string if needed
    if (
        hasattr(original_class, "__doc__")
        and isinstance(original_class.__doc__, str)
        and "n_jobs : int" not in original_class.__doc__
    ):
        parameters_doc_tail = "\n    Attributes"
        n_jobs_doc = """
    n_jobs : int, default=None
        The number of jobs to use in parallel for the computation.
        ``None`` means using all physical cores
        unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all logical cores.
        See :term:`Glossary <n_jobs>` for more details.
"""
        original_class.__doc__ = original_class.__doc__.replace(
            parameters_doc_tail, n_jobs_doc + parameters_doc_tail
        )

    return original_class


# Note: getting controller in global scope of this module is required
# to avoid overheads by its initialization per each function call
threadpool_controller = threadpoolctl.ThreadpoolController()


def get_suggested_n_threads(n_cpus):
    """Function to get `n_threads` limit
    if `n_jobs` is set in upper parallelization context.
    Usually, limit is equal to `n_logical_cpus` // `n_jobs`.
    Returns None if limit is not set.
    """
    n_threads_map = {
        lib_ctl.internal_api: lib_ctl.get_num_threads()
        for lib_ctl in threadpool_controller.lib_controllers
        if lib_ctl.internal_api != "mkl"
    }
    # openBLAS is limited by 128 threads by default.
    # thus, 128 threads from openBLAS is uninformative
    if "openblas" in n_threads_map and n_threads_map["openblas"] == 128:
        del n_threads_map["openblas"]
    # remove default values equal to n_cpus as uninformative
    for backend in list(n_threads_map.keys()):
        if n_threads_map[backend] == n_cpus:
            del n_threads_map[backend]
    if len(n_threads_map) > 0:
        return min(n_threads_map.values())
    else:
        return None


def run_with_n_jobs(method):
    """Decorator for running of methods containing oneDAL kernels with 'n_jobs'"""

    @wraps(method)
    def method_wrapper(self, *args, **kwargs):
        # threading parallel backend branch
        if not isinstance(threading.current_thread(), threading._MainThread):
            warn(
                "'Threading' parallel backend is not supported by "
                "Intel(R) Extension for Scikit-learn*. "
                "Falling back to usage of all available threads."
            )
            result = method(self, *args, **kwargs)
            return result
        # multiprocess parallel backends branch
        cl = self.__class__
        method_name = ".".join([cl.__module__, cl.__name__, method.__name__])
        # search for specified n_jobs
        n_jobs = self.n_jobs
        n_cpus = cpu_count()
        # receive n_threads limitation from upper parallelism context
        # using `threadpoolctl.ThreadpoolController`
        n_threads = get_suggested_n_threads(n_cpus)
        # get real `n_jobs` number of threads for oneDAL
        # using sklearn rules and `n_threads` from upper parallelism context
        if n_jobs is None or n_jobs == 0:
            if n_threads is None:
                # default branch with no setting for n_jobs
                return method(self, *args, **kwargs)
            else:
                n_jobs = n_threads
        elif n_jobs < 0:
            if n_threads is None:
                n_jobs = max(1, n_cpus + n_jobs + 1)
            else:
                n_jobs = max(1, n_threads + n_jobs + 1)
        # branch with set n_jobs
        old_n_threads = get_n_threads()
        if n_jobs != old_n_threads:
            logger = logging.getLogger("sklearnex")
            logger.debug(
                f"{method_name}: setting {n_jobs} threads (previous - {old_n_threads})"
            )
            set_n_threads(n_jobs)
        result = method(self, *args, **kwargs)
        if n_jobs != old_n_threads:
            set_n_threads(old_n_threads)
        return result

    return method_wrapper
