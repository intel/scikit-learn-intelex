# ==============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import onedal

from .dataframe_protocol import Column
from .dtype_conversion import get_data_type
from ..array import to_array, is_array_entity

data_type = onedal._backend.dtype
make_chunked_array = onedal._backend.data_management.make_chunked_array

class ChunkedArrayBuilder:
    def __init__(self, dtype: data_type):
        self.__dtype = dtype
        self.chunks = list()

    def append(self, chunk):
        assert is_array_entity(chunk)
        self.chunks.append(chunk)
        return self
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def dtype(self):
        return self.__dtype
    
    def __validate(self, result):
        count = result.get_chunk_count()
        assert count == self.chunk_count
        assert result.validate()

    def build(self):
        dtype, count = self.dtype, self.chunk_count
        result = make_chunked_array(dtype, count)

        for index in range(count):
            chunk = self.chunks[index]
            array = to_array(chunk)
            result.set_chunk(index, array)

        self.__validate(result)
        return result

def build_from_column(column: Column):
    dtype = get_data_type(column.dtype)
    builder = ChunkedArrayBuilder(dtype)

    for chunk in column.get_chunks():
        assert chunk.num_chunks() == 1
        buffers = chunk.get_buffers()
        raw_chunk, _ = buffers["data"] 
        builder.append(raw_chunk)

    return builder.build()
