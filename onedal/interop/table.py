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

import onedal

onedal_table = onedal._backend.data_management.table

from .empty_table import from_empty_table, is_empty_entity, to_empty_table
from .homogen_table import from_homogen_table, is_homogen_entity, to_homogen_table
from .sparse_table import from_sparse_table, is_sparse_entity, to_sparse_table


def is_table_entity(entity) -> bool:
    conditions = [is_empty_entity, is_sparse_entity, is_homogen_entity]
    return any(map(lambda check: check(entity), conditions))


def to_table(entity):
    assert is_table_entity(entity)
    if is_empty_entity(entity):
        result = to_empty_table(entity)
    elif is_sparse_entity(entity):
        result = to_sparse_table(entity)
    elif is_homogen_entity(entity):
        result = to_homogen_table(entity)
    else:
        raise ValueError("Not a known structure")
    result = onedal_table(result)
    assert is_table_entity(result)
    return result


# TODO: Make sure that it will return
# SYCL-native entity in the future
# Note: Check `sua_utils.py`
def from_table(table):
    assert is_table_entity(table)
    if is_empty_entity(table):
        result = from_empty_table(table)
    elif is_sparse_entity(table):
        result = from_sparse_table(table)
    elif is_homogen_entity(table):
        result = from_homogen_table(table)
    else:
        raise ValueError("Not a known table")
    assert is_table_entity(result)
    return result
