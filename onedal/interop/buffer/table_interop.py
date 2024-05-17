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

from .buffer_utils import is_buffer_entity, is_nd

table_kind = onedal._backend.data_management.table_kind
homogen_table = onedal._backend.data_management.homogen_table

wrap_to_homogen_table = onedal._backend.interop.buffer.wrap_to_homogen_table
wrap_from_homogen_table = onedal._backend.interop.buffer.wrap_from_homogen_table


def is_buffer_table(entity) -> bool:
    if is_buffer_entity(entity):
        return is_nd(entity, n=2)
    return False


def to_homogen_table(entity):
    assert is_buffer_table(entity)
    return wrap_to_homogen_table(entity)


def from_homogen_table(table) -> np.ndarray:
    assert table.get_kind() == table_kind.homogen
    assert isinstance(table, homogen_table)
    buffer = wrap_from_homogen_table(table)
    assert is_buffer_table(buffer)
    return np.asarray(buffer)
