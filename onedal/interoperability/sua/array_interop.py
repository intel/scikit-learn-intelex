import onedal

from .sua_utils import is_sua_entity, convert_sua, is_nd

wrap_to_array = onedal._backend.interop.sua.wrap_to_array
wrap_from_array = onedal._backend.interop.sua.wrap_from_array

def is_sua_array(entity) -> bool:
    if is_sua_entity(entity):
        return is_nd(entity, n=1)
    return False
    
def to_array(entity):
    assert is_sua_array(entity)
    return wrap_to_array(entity)

class fake_sua_array:
    def __init__(self, array):
        self.array = array

    @property
    def __sycl_usm_array_interface__(self) -> dict:
        if not hasattr(self, "sua") or self.sua is None:
            self.sua: dict = wrap_from_array(self.array)
        return self.sua

def from_array(array):
    result = fake_sua_array(array)
    assert is_sua_array(result)
    return convert_sua(result)
