import onedal
import numpy as np

from .buffer_utils import is_buffer_entity, is_nd

table_kind = onedal._backend.data_management.table_kind
homogen_table = onedal._backend.data_management.homogen_table

wrap_to_homogen_table = onedal._backend.interop.buffer.wrap_to_homogen_table
wrap_from_homogen_table = onedal._backend.interop.buffer.wrap_from_homogen_table

def is_buffer_table(entity) -> bool:
    if is_buffer_entity(entity):
        return is_nd(entity, n = 2)
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
