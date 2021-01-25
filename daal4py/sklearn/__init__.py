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

from .monkeypatch.dispatcher import enable as patch_sklearn
from .monkeypatch.dispatcher import disable as unpatch_sklearn
from .monkeypatch.dispatcher import _patch_names as sklearn_patch_names
from .monkeypatch.dispatcher import _get_map_of_algorithms as sklearn_patch_map

__all__ = [
    "patch_sklearn", "unpatch_sklearn", "sklearn_patch_names",
    "sklearn_patch_map", "cluster", "decomposition", "ensemble",
    "linear_model", "manifold", "neighbors",
    "svm", "tree", "utils", "model_selection", "metrics",
]
