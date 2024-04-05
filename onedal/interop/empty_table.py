# ==============================================================================
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
# ==============================================================================

import numpy as np

import onedal

onedal_table = onedal._backend.data_management.table


def is_python_empty(entity) -> bool:
    return entity is None


def is_native_empty(entity) -> bool:
    if hasattr(entity, "has_data"):
        return not entity.has_data()
    return False


def is_empty_entity(entity) -> bool:
    conditions = [is_native_empty, is_python_empty]
    return any(map(lambda check: check(entity), conditions))


def to_empty_table(entity):
    assert is_empty_entity(entity)
    return onedal_table()


def from_empty_table(table):
    assert is_empty_entity(table)
    return np.array([], dtype=np.float32)
