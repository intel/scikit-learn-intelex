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

from functools import lru_cache

from .dataframe_protocol import DtypeKind, Dtype, CategoricalDescription

data_type = onedal._backend.dtype
feature_type = onedal._backend.ftype

@lru_cache
def get_dtype_map() -> dict:
    return {
        (DtypeKind.INT, 8): data_type.int8,
        (DtypeKind.INT, 16): data_type.int16,
        (DtypeKind.INT, 32): data_type.int32,
        (DtypeKind.INT, 64): data_type.int64,
        (DtypeKind.UINT, 8): data_type.uint8,
        (DtypeKind.UINT, 16): data_type.uint16,
        (DtypeKind.UINT, 32): data_type.uint32,
        (DtypeKind.UINT, 64): data_type.uint64,
        (DtypeKind.FLOAT, 32): data_type.float32,
        (DtypeKind.FLOAT, 64): data_type.float64
    }

@lru_cache
def get_ftype_map() -> dict:
    return {
        (True, True): data_type.ordinal,
        (True, False): feature_type.ratio,
        (False, True): feature_type.nominal,
        (False, False): feature_type.interval
    }


def get_data_type(dtype: Dtype):
    kind, bits, _, _ = dtype
    type_desc = (kind, bits)
    dtype_map = get_dtype_map()
    return dtype_map[type_desc]

def get_feature_type(ftype: CategoricalDescription):
    ftype_map = get_ftype_map()
    is_ordered = ftype["is_ordered"]
    is_dictionary = ftype["is_dictionary"]
    type_desc = (is_ordered, is_dictionary)
    return ftype_map[type_desc]
