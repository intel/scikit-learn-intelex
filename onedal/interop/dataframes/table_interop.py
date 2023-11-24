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

from .dataframe_builder import build_from_dataframe

table = onedal._backend.data_management.table
heterogen_table = onedal._backend.data_management.heterogen_table
heterogen_kind = onedal._backend.data_management.table_kind.heterogen

dataframe = "__dataframe__"
dataframe_standard = "__dataframe_standard__"
dataframe_namespace = "__dataframe_namespace__"


def is_python_dataframe(entity) -> bool:
    namespace = hasattr(entity, dataframe_namespace)
    namespace = namespace or hasattr(entity, dataframe)
    return namespace or hasattr(entity, dataframe_standard)

def is_native_dataframe(entity) -> bool:
    if isinstance(entity, table):
        kind = entity.get_kind()
        return heterogen_kind == kind
    return isinstance(entity, heterogen_table)

def is_dataframe_entity(entity) -> bool:
    conditions = [is_native_dataframe, is_python_dataframe]
    return any(map(lambda check: check(entity), conditions))

def to_heterogen_table_python(entity):
    assert is_python_dataframe(entity)
    if hasattr(entity, dataframe):
        temp = entity.__dataframe__()
    elif hasattr(entity, dataframe_standard):
        temp = entity.__dataframe_standard__()
    elif hasattr(entity, dataframe_namespace):
        temp = entity
    else:
        raise TypeError("Expected DataFrame")
    return build_from_dataframe(temp)

def to_heterogen_table_native(entity):
    assert is_native_dataframe(entity)
    return build_from_dataframe(entity)

def to_table(entity) -> table:
    assert is_dataframe_entity(entity)
    if is_native_dataframe(entity):
        result = to_heterogen_table_native(entity)
    elif is_python_dataframe(entity):
        result = to_heterogen_table_python(entity)
    else:
        raise ValueError("Unknown kind of dataframe")
    assert is_native_dataframe(result)
    return result
