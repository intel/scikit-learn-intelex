from typing import Callable
from onedal.interoperability.utils import check_attr

try:
    import dpctl.tensor as dpt

    # TODO: Make sure that it will return
    # SYCL-native entity in the future
    def __convert_sua_impl(sua):
        result = dpt.asarray(sua)
        return dpt.asnumpy(result)
    
    convert_sua_impl = __convert_sua_impl
except ImportError:
    convert_sua_impl = None

def is_valid_sua_iface(iface: dict) -> bool:
    if isinstance(iface, dict):
        check_version: Callable = lambda v: v == 1
        return check_attr(iface, "version", check_version)
    return False

def is_sua_entity(entity) -> bool:
    iface_attr: str = "__sycl_usm_array_interface__"
    if hasattr(entity, iface_attr):
        iface: dict = getattr(entity, iface_attr)
        return is_valid_sua_iface(iface)
    return False

def get_sua_iface(entity) -> dict:
    assert is_sua_entity(entity)
    return entity.__sycl_usm_array_interface__

# TODO: Make sure that it will return
# SYCL-native entity in the future
def convert_sua(sua):
    assert is_sua_entity(sua)
    if convert_sua_impl is not None:
        return convert_sua_impl(sua)
    return None

def is_nd(entity, n: int = 1) -> bool:
    if is_sua_entity(entity):
        iface = get_sua_iface(entity)
        shape = iface["shape"]
        return len(shape) == n
    return False
