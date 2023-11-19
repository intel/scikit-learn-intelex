import onedal
import numpy as np

from .array import to_array, from_array

from scipy.sparse import isspmatrix_csr, csr_matrix

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

def assert_table(table, matrix):
    assert is_csr_entity(matrix)
    row_count, col_count = matrix.shape
    assert table.get_non_zero_count() == matrix.getnnz()
    assert table.get_column_count() == col_count
    assert table.get_row_count() == row_count

# Passing throw native entity
def to_csr_table_native(entity) -> csr_table:
    assert is_native_csr(entity)
    return csr_table(entity)

def to_typed_array(x, dtypes = [np.int64]):
    result = x
    if x.dtype not in list(dtypes):
        result = x.astype(dtypes[0])
    assert result.dtype in dtypes
    return to_array(result)

# Converting python entity to table
# TODO: Implement smarter logic
def to_csr_table_python(entity) -> csr_table:
    assert isspmatrix_csr(entity)
    _, col_count = entity.shape
    ids = to_typed_array(entity.indices)
    print(ids.get_policy())
    ofs = to_typed_array(entity.indptr)
    print(ofs.get_policy())
    nz = to_array(entity.data)
    print(nz.get_policy())
    result = csr_table(nz, ids, ofs, \
        col_count, sparse_indexing.zero_based)
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
    col_count = table.get_column_count()
    row_count = table.get_row_count()
    shape = (row_count, col_count)
    ids = from_array(table.get_column_indices())
    ofs = from_array(table.get_row_offsets())
    nz = from_array(table.get_data())
    data = (nz, ids, ofs)
    result = csr_matrix(data, shape)
    assert_table(table, result)
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
