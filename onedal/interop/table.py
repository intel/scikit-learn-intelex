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

from .csr_table import is_csr_entity, from_csr_table, to_csr_table
from .empty_table import is_empty_entity, from_empty_table, to_empty_table
from .homogen_table import is_homogen_entity, from_homogen_table, to_homogen_table


def is_table_entity(entity) -> bool:
    conditions = [is_empty_entity, is_csr_entity, is_homogen_entity]
    return any(map(lambda check: check(entity), conditions))


def to_table(entity):
    assert is_table_entity(entity)
    if is_empty_entity(entity):
        result = to_empty_table(entity)
    elif is_csr_entity(entity):
        result = to_csr_table(entity)
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
    elif is_csr_entity(table):
        result = from_csr_table(table)
    elif is_homogen_entity(table):
        result = from_homogen_table(table)
    else:
        raise ValueError("Not a known table")
    assert is_table_entity(result)
    return result
