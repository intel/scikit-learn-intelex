import onedal

get_shape = onedal._backend.interop.dlpack.get_shape

def is_valid_dlpack_capsule(caps) -> bool:
    caps_repr: str = str(caps)
    type_repr: str = str(type(caps))
    if "PyCapsule" in type_repr:
        if "dltensor" in caps_repr:
            return True
    return False

def get_dlpack_capsule(entity):
    iface_attr: str = "__dlpack__"
    if hasattr(entity, iface_attr):
        gen = getattr(entity, iface_attr)
        return gen()
    return None 

def is_dlpack_entity(entity) -> bool:
    caps = get_dlpack_capsule(entity)
    if caps is not None:
        return is_valid_dlpack_capsule(caps)
    else:
        return False

def is_nd(entity, n: int = 1) -> bool:
    if is_dlpack_entity(entity):
        caps = get_dlpack_capsule(entity)
        shape = get_shape(caps)
        return len(shape) == n
    return False
