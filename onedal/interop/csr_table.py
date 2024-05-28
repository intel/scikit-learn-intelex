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
from scipy.sparse import csr_matrix, isspmatrix_csr

import onedal

from .array import from_array, to_array, to_common_policy, to_typed_array

table = onedal._backend.data_management.table
csr_table = onedal._backend.data_management.csr_table
csr_kind = onedal._backend.data_management.table_kind.csr
sparse_indexing = onedal._backend.data_management.sparse_indexing


def is_native_csr(entity) -> bool:
    if isinstance(entity, table):
        kind = entity.get_kind()
        return kind == csr_kind
    return isinstance(entity, csr_table)


def is_csr_entity(entity) -> bool:
    is_native = is_native_csr(entity)
    is_scipy = isspmatrix_csr(entity)
    return is_native or is_scipy


def assert_table(entity, matrix):
    assert is_native_csr(entity)
    assert isspmatrix_csr(matrix)
    row_count, col_count = matrix.shape
    assert entity.get_column_count() == col_count
    assert entity.get_row_count() == row_count


# Passing throw native entity
def to_csr_table_native(entity) -> csr_table:
    assert is_native_csr(entity)
    return csr_table(entity)


# Converting python entity to table
# TODO #1: Implement smarter logic for
# type conversion & device support
# TODO #2: Implement with `zero_based`
# indexing, it is a workaround for now.
# The glue logic between oneDAL & DAAL is
# suspicious for having bug with `zero_based`
def to_csr_table_python(entity) -> csr_table:
    assert isspmatrix_csr(entity)
    row_count, col_count = entity.shape

    def to_indices(arr, ids=[np.int64]):
        return to_typed_array(arr, ids)

    ids = to_indices(entity.indices + 1)
    ofs = to_indices(entity.indptr + 1)
    nz = to_array(entity.data)

    one_based = sparse_indexing.one_based
    ids, ofs, nz = to_common_policy(ids, ofs, nz)
    result = csr_table(nz, ids, ofs, col_count, one_based)
    assert col_count == result.get_column_count()
    assert row_count == result.get_row_count()
    assert_table(result, entity)
    return result


def to_csr_table(entity) -> csr_table:
    assert is_csr_entity(entity)
    if is_native_csr(entity):
        result = to_csr_table_native(entity)
    elif isspmatrix_csr(entity):
        result = to_csr_table_python(entity)
    else:
        raise ValueError("Not able to convert")
    assert is_native_csr(result)
    return result


# Converting onedal entity to python
def from_csr_table_native(entity) -> csr_matrix:
    assert is_native_csr(entity)
    entity = to_csr_table_native(entity)
    col_count = entity.get_column_count()
    row_count = entity.get_row_count()
    shape = (row_count, col_count)
    ids = from_array(entity.get_column_indices())
    ofs = from_array(entity.get_row_offsets())
    nz = from_array(entity.get_data())
    indexing = entity.get_indexing()
    zb = sparse_indexing.zero_based
    ob = sparse_indexing.one_based

    if indexing == ob:
        ids, ofs = ids - 1, ofs - 1
    else:
        assert indexing == zb

    data = (nz, ids, ofs)
    result = csr_matrix(data, shape)
    assert_table(entity, result)
    return result


# Passing matrix that is already in correct format
def from_csr_table_python(entity) -> csr_matrix:
    assert isspmatrix_csr(entity)
    return entity


def from_csr_table(table) -> csr_matrix:
    if is_native_csr(table):
        csr = csr_table(table)
        result = from_csr_table_native(csr)
    elif isspmatrix_csr(table):
        result = from_csr_table_python(table)
    else:
        raise ValueError("Not able to convert")
    assert isspmatrix_csr(result)
    return result
