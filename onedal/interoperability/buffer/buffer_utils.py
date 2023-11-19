import onedal

get_shape = onedal._backend.interop.buffer.get_shape
get_strides = onedal._backend.interop.buffer.get_strides

def is_buffer_entity(entity) -> bool:
    try:
        memoryview(entity)
        return True
    except TypeError:
        return False
    
def is_nd(entity, n: int = 1) -> bool:
    if is_buffer_entity(entity):
        shape = get_shape(entity)
        return len(shape) == n
    return False
