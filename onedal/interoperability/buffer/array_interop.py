import onedal
import numpy as np

from .buffer_utils import is_buffer_entity, is_nd

wrap_to_array = onedal._backend.interop.buffer.wrap_to_array
wrap_from_array = onedal._backend.interop.buffer.wrap_from_array

# TODO: implement more complex logic of
# checking shape & strides in entity
def is_buffer_array(entity) -> bool:
    if is_buffer_entity(entity):
        return is_nd(entity, n = 1)
    return False
    
def to_array(entity):
    assert is_buffer_array(entity)
    return wrap_to_array(entity)

def from_array(array) -> np.ndarray:
    buffer = wrap_from_array(array)
    assert is_buffer_array(buffer)
    return np.asarray(buffer)
