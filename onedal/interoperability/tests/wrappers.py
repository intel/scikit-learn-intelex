from array import array
from functools import lru_cache

dtype_map = {
    'int8': 'b',
    'uint8': 'B',
    'int16': 'h',
    'uint16': 'H',
    'int32': 'i',
    'uint32': 'I',
    'int64': 'q',
    'uint64': 'Q',
    'float32': 'f',
    'float64': 'd'
}

@lru_cache
def get_dtype_list():
    return list(dtype_map.keys())

class only_sua_wrapper:
    def __init__(self, body):
        self.body = body

    @property
    def __sycl_usm_array_interface__(self):
        return self.body.__sycl_usm_array_interface__

class only_dlpack_wrapper:
    def __init__(self, body):
        self.body = body

    def __dlpack__(self):
        return self.body.__dlpack__()

    def __dlpack_device__(self):
        return self.body.__dlpack_device__()

# It is sthe easiest way to get a purely
# buffer entity. It may (or may not)
# preserve pointers and also don't
# support multidimensional arrays
def only_buffer_wrapper(body):
    data = body.tobytes()
    dtype = body.dtype.name
    dtype = dtype_map[dtype]
    return array(dtype, data)

def wrap_entity(entity, backend):
    if backend == "sua":
        return only_sua_wrapper(entity)
    elif backend == "buffer":
        return only_buffer_wrapper(entity)
    elif backend == "dlpack":
        return only_dlpack_wrapper(entity)
    else:
        return entity
