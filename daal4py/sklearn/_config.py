# ==============================================================================
# Copyright 2024 Intel Corporation
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

"""Tools to expose sklearnex's config settings to daal4py level."""

import threading

_default_global_config = {
    "target_offload": "auto",
    "allow_fallback_to_host": False,
}

_threadlocal = threading.local()


def _get_daal4py_threadlocal_config():
    if not hasattr(_threadlocal, "d4p_global_config"):
        _threadlocal.d4p_global_config = _default_global_config.copy()
    return _threadlocal.d4p_global_config


def _get_config():
    """Retrieve current values for configuration set by :func:`set_config`
    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    See Also
    --------
    _set_config : Set global configuration.
    """
    daal4py_config = _get_daal4py_threadlocal_config().copy()
    return {**daal4py_config}


def _set_config(target_offload=None, allow_fallback_to_host=None):
    """Set global configuration
    Parameters
    ----------
    target_offload : string or dpctl.SyclQueue, default=None
        The device primarily used to perform computations.
        If string, expected to be "auto" (the execution context
        is deduced from input data location),
        or SYCL* filter selector string. Global default: "auto".
    allow_fallback_to_host : bool, default=None
        If True, allows to fallback computation to host device
        in case particular estimator does not support the selected one.
        Global default: False.
    See Also
    --------
    _get_config : Retrieve current values of the global configuration.
    """

    local_config = _get_daal4py_threadlocal_config()

    if target_offload is not None:
        local_config["target_offload"] = target_offload
    if allow_fallback_to_host is not None:
        local_config["allow_fallback_to_host"] = allow_fallback_to_host
