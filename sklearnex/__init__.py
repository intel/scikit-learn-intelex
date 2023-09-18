#!/usr/bin/env python
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

from . import utils
from ._config import config_context, get_config, set_config
from .dispatcher import (
    get_patch_map,
    get_patch_names,
    is_patched_instance,
    patch_sklearn,
    sklearn_is_patched,
    unpatch_sklearn,
)

__all__ = [
    "basic_statistics",
    "cluster",
    "config_context",
    "decomposition",
    "ensemble",
    "get_config",
    "get_patch_map",
    "get_patch_names",
    "is_patched_instance",
    "linear_model",
    "manifold",
    "metrics",
    "neighbors",
    "patch_sklearn",
    "set_config",
    "sklearn_is_patched",
    "sklearn_is_patchedget_patch_map",
    "svm",
    "unpatch_sklearn",
    "utils",
]


from onedal import _is_dpc_backend

if _is_dpc_backend:
    __all__.append("spmd")


from ._utils import set_sklearn_ex_verbose

set_sklearn_ex_verbose()
