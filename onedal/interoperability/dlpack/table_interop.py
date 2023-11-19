import onedal

from .dlpack_utils import is_dlpack_entity, is_nd

wrap_to_homogen_table = onedal._backend.interop.dlpack.wrap_to_homogen_table

# TODO: implement more complex logic of
# checking shape & strides in entity
def is_dlpack_table(entity) -> bool:
    if is_dlpack_entity(entity):
        return is_nd(entity, n = 2)
    return False
    
def to_homogen_table(entity):
    assert is_dlpack_table(entity)
    return wrap_to_homogen_table(entity)
