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
import pytest
from scipy.sparse import csr_matrix, find, isspmatrix_csr

import onedal
from onedal.interop.array import from_array
from onedal.interop.csr_table import from_csr_table, is_csr_entity, to_csr_table

from .wrappers import get_dtype_list

table_dimensions = [
    (1, 1),
    (1, 11),
    (17, 1),
    (3, 7),
    (21, 47),
    (1001, 1),
    (999, 1),
    (888, 777),
    (123, 999),
]


# Used instead of `scipy.sparse.rand` due to the
# reasons:
# 1. Different types of integer indices
# 2. Zero samples in case of small matrices
# 3. More checks for the content
def generate_csr_data(gen, shape, per_row, dtypes):
    row_count, col_count = shape
    data_dtype, index_dtype = dtypes
    min_per_row, max_per_row = per_row

    assert 1 <= min_per_row
    assert min_per_row <= max_per_row
    assert max_per_row <= col_count + 1

    max_value = min_per_row * max_per_row

    data_chunks, index_chunks, counts = [], [], [0]
    for _ in range(row_count):
        counts.append(gen.integers(min_per_row, max_per_row))
        data_chunks.append(gen.integers(0, max_value, counts[-1]))
        index_chunks.append(gen.integers(0, col_count, counts[-1]))

    assert len(counts) == row_count + 1
    assert len(data_chunks) == row_count
    assert len(index_chunks) == row_count

    offsets = np.cumsum(counts).astype(index_dtype)
    data = np.concatenate(data_chunks).astype(data_dtype)
    indices = np.concatenate(index_chunks).astype(index_dtype)

    del counts, data_chunks, index_chunks

    nnz = len(data)
    assert offsets[-1] == nnz
    assert len(indices) == nnz
    assert min_per_row * row_count <= nnz
    assert nnz <= max_per_row * row_count
    assert len(offsets) == row_count + 1

    return (data, indices, offsets)


sp_indexing = onedal._backend.data_management.sparse_indexing
indexing_offset_map = {sp_indexing.zero_based: 0, sp_indexing.one_based: 1}


@pytest.mark.parametrize("shape", table_dimensions)
@pytest.mark.parametrize("dtype", get_dtype_list())
@pytest.mark.parametrize("itype", [np.int32, np.uint32, np.int64])
def test_host_csr_table_functionality(shape, dtype, itype):
    row_count, col_count = shape
    min_per_row = max(1, col_count // 10)
    max_per_row = min(2 * min_per_row, col_count) + 1
    per_row, dtypes = (min_per_row, max_per_row), (dtype, itype)
    generator = np.random.Generator(np.random.MT19937(sum(shape)))
    components = generate_csr_data(generator, shape, per_row, dtypes)
    scipy_data, scipy_indices, scipy_offsets = components

    scipy_table = csr_matrix(components, shape)
    assert isspmatrix_csr(scipy_table)
    assert is_csr_entity(scipy_table)

    onedal_table = to_csr_table(scipy_table)
    assert is_csr_entity(onedal_table)

    assert onedal_table.get_row_count() == row_count
    assert onedal_table.get_column_count() == col_count

    curr_indexing = onedal_table.get_indexing()
    offset = indexing_offset_map[curr_indexing]

    def get_indices(array):
        raw = from_array(array)
        return raw - offset

    onedal_indices = get_indices(onedal_table.get_column_indices())
    np.testing.assert_equal(scipy_indices, onedal_indices)
    onedal_offsets = get_indices(onedal_table.get_row_offsets())
    np.testing.assert_equal(scipy_offsets, onedal_offsets)
    onedal_data = from_array(onedal_table.get_data())
    np.testing.assert_equal(scipy_data, onedal_data)

    del onedal_indices, onedal_offsets, onedal_data

    return_table = from_csr_table(onedal_table)
    dv, dr, dc = find(return_table - scipy_table)
    assert len(dv) == 0 and len(dr) == 0 and len(dc) == 0
