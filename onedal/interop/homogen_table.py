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
import onedal.interop.buffer as buffer
import onedal.interop.dlpack as dlpack
import onedal.interop.sua as sua
from onedal.interop.utils import is_host_policy

from .array import is_array_entity, to_array

table = onedal._backend.data_management.table
homogen_table = onedal._backend.data_management.homogen_table
homogen_kind = onedal._backend.data_management.table_kind.homogen


def is_native_homogen(entity) -> bool:
    if isinstance(entity, table):
        kind = entity.get_kind()
        return kind == homogen_kind
    return isinstance(entity, homogen_table)


def is_python_homogen(entity) -> bool:
    conditions = [sua.is_sua_table, dlpack.is_dlpack_table, buffer.is_buffer_table]
    return any(map(lambda check: check(entity), conditions))


def is_homogen_entity(entity) -> bool:
    conditions = [is_native_homogen, is_python_homogen, is_array_entity]
    return any(map(lambda check: check(entity), conditions))


def to_homogen_table_array(entity) -> homogen_table:
    assert is_array_entity(entity)
    entity = to_array(entity)
    count = entity.get_count()
    return homogen_table(entity, count, 1)


def to_homogen_table_native(entity) -> homogen_table:
    assert is_native_homogen(entity)
    return homogen_table(entity)


def to_homogen_table_python(entity) -> homogen_table:
    assert is_python_homogen(entity)
    if sua.is_sua_table(entity):
        return sua.to_homogen_table(entity)
    elif dlpack.is_dlpack_table(entity):
        return dlpack.to_homogen_table(entity)
    elif buffer.is_buffer_table(entity):
        return buffer.to_homogen_table(entity)
    else:
        raise ValueError("Not a python homogen table")


def to_homogen_table(entity) -> homogen_table:
    assert is_homogen_entity(entity)
    if is_array_entity(entity):
        result = to_homogen_table_array(entity)
    elif is_native_homogen(entity):
        result = to_homogen_table_native(entity)
    elif is_python_homogen(entity):
        result = to_homogen_table_python(entity)
    else:
        raise ValueError("Not a homogen table")
    assert is_native_homogen(result)
    return result


def from_homogen_table_native(table):
    assert is_native_homogen(table)
    data = table.get_data()
    policy = data.get_policy()
    device = policy.get_device_name()
    if is_host_policy(policy):
        return buffer.from_homogen_table(table)
    elif device in ["cpu", "gpu"]:
        return sua.from_homogen_table(table)
    else:
        raise ValueError("unknown device")


def from_homogen_table_python(table):
    assert is_python_homogen(table)
    return table


def from_homogen_table(table):
    assert is_homogen_entity(table)
    if is_native_homogen(table):
        homogen = homogen_table(table)
        result = from_homogen_table_native(homogen)
    elif is_python_homogen(table):
        result = from_homogen_table_python(table)
    else:
        raise ValueError("Not able to convert from homogen table")
    assert is_python_homogen(result)
    return result
