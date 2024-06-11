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

"""Tools to expose some sklearnex's config settings to onedal4py level."""

import threading

_default_global_config = {
    "target_offload": "auto",
    "allow_fallback_to_host": False,
}

_threadlocal = threading.local()


def _get_onedal_threadlocal_config():
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _default_global_config.copy()
    return _threadlocal.global_config


# TODO:
# docstrings
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
    onedal_config = _get_onedal_threadlocal_config().copy()
    return {**onedal_config}
