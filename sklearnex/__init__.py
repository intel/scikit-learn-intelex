#!/usr/bin/env python
#===============================================================================
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
#===============================================================================

from .dispatcher import patch_sklearn
from .dispatcher import unpatch_sklearn
from .dispatcher import get_patch_names
from .dispatcher import get_patch_map
from ._config import get_config, set_config, config_context

__all__ = [
    "patch_sklearn", "unpatch_sklearn", "get_patch_names",
    "get_patch_map", "get_config", "set_config", "config_context",
    "cluster", "decomposition", "ensemble", "linear_model",
    "manifold", "neighbors", "svm", "metrics", "utils"
]

from ._utils import set_sklearn_ex_verbose

set_sklearn_ex_verbose()
