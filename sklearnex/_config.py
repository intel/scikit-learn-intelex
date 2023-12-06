# ==============================================================================
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
# ==============================================================================

import os
import threading
from contextlib import contextmanager

from sklearn import get_config as skl_get_config
from sklearn import set_config as skl_set_config

from onedal import _backend, _is_dpc_backend

if _is_dpc_backend:
    ComputeMode = _backend.ComputeMode

_default_global_config = {
    "target_offload": "auto",
    "allow_fallback_to_host": False,
    "compute_mode": os.environ.get("DAL_BLAS_COMPUTE_MODE"),
}

_threadlocal = threading.local()


def _get_sklearnex_threadlocal_config():
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _default_global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`
    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    See Also
    --------
    config_context : Context manager for global configuration.
    set_config : Set global configuration.
    """
    sklearn = skl_get_config()
    sklearnex = _get_sklearnex_threadlocal_config().copy()
    sklearnex["compute_mode"] = os.environ.get("DAL_BLAS_COMPUTE_MODE")
    return {**sklearn, **sklearnex}


def set_config(
    target_offload=None, allow_fallback_to_host=None, compute_mode=None, **sklearn_configs
):
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
    compute_mode : str or list, default=None
        The computational method used for BLAS level 3 routines in
        oneDAL for GPU devices. The compute_mode string(s) must be
        from the oneMKL compute_mode enum list. None will not
        change the set compute_mode.
        Global default: os environment variable DAL_BLAS_COMPUTE_MODE
    See Also
    --------
    config_context : Context manager for global configuration.
    get_config : Retrieve current values of the global configuration.
    """
    skl_set_config(**sklearn_configs)

    local_config = _get_sklearnex_threadlocal_config()

    if target_offload is not None:
        local_config["target_offload"] = target_offload
    if allow_fallback_to_host is not None:
        local_config["allow_fallback_to_host"] = allow_fallback_to_host
    if _is_dpc_backend:
        if compute_mode is not None:
            try:
                inp = (
                    compute_mode.split(",") if type(compute_mode) is str else compute_mode
                )
                os.environ["DAL_BLAS_COMPUTE_MODE"] = ",".join(
                    [ComputeMode[i.lower()].name for i in inp]
                ).upper()
            except (KeyError, AttributeError) as e:
                raise ValueError(
                    f"'{e.args[0]}' is not a supported compute_mode"
                ) from None
        else:
            os.environ.pop("DAL_BLAS_COMPUTE_MODE", None)


@contextmanager
def config_context(**new_config):
    """Context manager for global scikit-learn configuration
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
    compute_mode : str or list, default=None
        The computational method used for BLAS level 3 routines in
        oneDAL for GPU devices. The compute_mode string(s) must be
        from the oneMKL compute_mode enum list. None will not
        change the set compute_mode.
        Global default: os environment variable DAL_BLAS_COMPUTE_MODE
    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.
    See Also
    --------
    set_config : Set global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.
    """
    old_config = get_config()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
