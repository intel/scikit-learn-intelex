# ===============================================================================
# Copyright 2023 Intel Corporation
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
# ===============================================================================

from daal4py.sklearn._utils import daal_check_version
from onedal import _backend


def get_tree_state_cls(model, iTree, n_classes):
    return _backend.get_tree.classification.get_tree_state(
        model, iTree, n_classes)


def get_tree_state_reg(model, iTree):
    return _backend.get_tree.regression.get_tree_state(model, iTree, 1)


if daal_check_version((2023, 'P', 301)):
    def get_forest_state(model, n_classes=None):
        if n_classes:
            return _backend.get_tree.classification.get_all_states(model, n_classes)
        else:
            return _backend.get_tree.regression.get_all_states(model, 1)