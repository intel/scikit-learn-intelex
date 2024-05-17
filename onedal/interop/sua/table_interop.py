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

from .sua_utils import convert_sua, is_nd, is_sua_entity

table_kind = onedal._backend.data_management.table_kind
homogen_table = onedal._backend.data_management.homogen_table
wrap_to_homogen_table = onedal._backend.interop.sua.wrap_to_homogen_table
wrap_from_homogen_table = onedal._backend.interop.sua.wrap_from_homogen_table


def is_sua_table(entity) -> bool:
    if is_sua_entity(entity):
        return is_nd(entity, n=2)
    else:
        return False


def to_homogen_table(entity):
    assert is_sua_table(entity)
    return wrap_to_homogen_table(entity)


class fake_sua_table:
    def __init__(self, table):
        self.table = table

    @property
    def __sycl_usm_array_interface__(self) -> dict:
        if not hasattr(self, "sua") or self.sua is None:
            self.sua: dict = wrap_from_homogen_table(self.table)
        return self.sua


def from_homogen_table(table):
    assert table.get_kind() == table_kind.homogen
    assert isinstance(table, homogen_table)
    result = fake_sua_table(table)
    assert is_sua_table(result)
    return convert_sua(result)
