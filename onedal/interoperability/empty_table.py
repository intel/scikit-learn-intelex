import onedal
import numpy as np

onedal_table = onedal._backend.data_management.table

def is_python_empty(entity) -> bool:
    return entity is None

def is_native_empty(entity) -> bool:
    if hasattr(entity, "has_data"):
        return not entity.has_data()
    return False

def is_empty_entity(entity) -> bool:
    conditions = [is_native_empty, is_python_empty]
    return any(map(lambda check: check(entity), conditions))

def to_empty_table(entity):
    assert is_empty_entity(entity)
    return onedal_table()

def from_empty_table(table):
    assert is_empty_entity(table)
    return np.array([], dtype = np.float32)
