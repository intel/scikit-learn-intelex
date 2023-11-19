import onedal

from .dlpack_utils import is_dlpack_entity, is_nd

wrap_to_array = onedal._backend.interop.dlpack.wrap_to_array

# TODO: implement more complex logic of
# checking shape & strides in entity
def is_dlpack_array(entity) -> bool:
    if is_dlpack_entity(entity):
        return is_nd(entity, n = 1)
    return False
    
def to_array(entity):
    assert is_dlpack_array(entity)
    return wrap_to_array(entity)
