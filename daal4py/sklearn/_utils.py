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
from functools import wraps


def set_idp_sklearn_verbose():
    import logging
    import warnings
    import os
    import sys
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
    import sys
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
        return find_common_type(X.dtypes) if is_DataFrame(X) else X.dtype
    except ImportError:
        return getattr(X, "dtype", None)


def get_number_of_types(dataframe):
    dtypes = getattr(dataframe, "dtypes", None)
    try:
        return len(set(dtypes))
    except TypeError:
        return 1


def _create_global_queue_from_sycl_context():
    import sys
    d4p_target = None
    if 'daal4py.oneapi' in sys.modules:
        from daal4py.oneapi import _get_device_name_sycl_ctxt
        d4p_target = _get_device_name_sycl_ctxt()

    if d4p_target is not None:
        try:
            from dpctl import SyclQueue
        except ImportError:
            raise RuntimeError("dpctl need to be installed for device offload")
        return SyclQueue(d4p_target if d4p_target != 'host' else 'cpu')
    return None


def _transfer_to_host(queue, *data):
    has_usm_data = False

    host_data = []
    for item in data:
        usm_iface = getattr(item, '__sycl_usm_array_interface__', None)
        if usm_iface is not None:
            import dpctl.memory as dp_mem
            import numpy as np

            if queue is not None:
                if queue.sycl_device != usm_iface['syclobj'].sycl_device:
                    raise RuntimeError('Input data shall be located '
                                       'on single target device')
            else:
                queue = usm_iface['syclobj']

            buffer = dp_mem.as_usm_memory(item).copy_to_host()
            item = np.ndarray(shape=usm_iface['shape'],
                              dtype=usm_iface['typestr'],
                              buffer=buffer)
            has_usm_data = True
        elif has_usm_data and item is not None:
            raise RuntimeError('Input data shall be located on single target device')
        host_data.append(item)
    return queue, host_data


def _copy_to_usm(queue, array):
    from dpctl.memory import MemoryUSMDevice
    from dpctl.tensor import usm_ndarray

    mem = MemoryUSMDevice(array.nbytes, queue=queue)
    mem.copy_from_host(array.tobytes())
    return usm_ndarray(array.shape, array.dtype, buffer=mem)


def _get_host_inputs(*args, **kwargs):
    import sys
    if 'sklearnex' in sys.modules:
        from sklearnex._device_offload import _get_global_queue as get_queue
    else:
        get_queue = _create_global_queue_from_sycl_context

    q = get_queue()
    q, hostargs = _transfer_to_host(q, *args)
    q, hostvalues = _transfer_to_host(q, *kwargs.values())
    hostkwargs = dict(zip(kwargs.keys(), hostvalues))
    return q, hostargs, hostkwargs


def _extract_usm_iface(*args, **kwargs):
    return getattr((*args, *kwargs.values())[0],
                   '__sycl_usm_array_interface__',
                   None)


def _run_on_device(func, queue, obj=None, *args, **kwargs):
    def dispatch_by_obj(obj, func, *args, **kwargs):
        if obj is not None:
            return func(obj, *args, **kwargs)
        return func(*args, **kwargs)

    if queue is not None:
        from daal4py.oneapi import sycl_context

        with sycl_context('gpu' if queue.sycl_device.is_gpu else 'cpu'):
            return dispatch_by_obj(obj, func, *args, **kwargs)
    return dispatch_by_obj(obj, func, *args, **kwargs)


def support_usm_ndarray(freefunc=False):
    def decorator(func):
        def wrapper_impl(obj, *args, **kwargs):
            usm_iface = _extract_usm_iface(*args, **kwargs)
            q, hostargs, hostkwargs = _get_host_inputs(*args, **kwargs)
            result = _run_on_device(func, q, obj, *hostargs, **hostkwargs)
            if usm_iface is not None and hasattr(result, '__array_interface__'):
                return _copy_to_usm(q, result)
            return result

        if freefunc:
            @wraps(func)
            def wrapper_free(*args, **kwargs):
                return wrapper_impl(None, *args, **kwargs)
            return wrapper_free
        else:
            @wraps(func)
            def wrapper_with_self(self, *args, **kwargs):
                return wrapper_impl(self, *args, **kwargs)
            return wrapper_with_self
    return decorator
