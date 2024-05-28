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

import numpy as np

import onedal

from .buffer_utils import is_buffer_entity, is_nd

wrap_to_array = onedal._backend.interop.buffer.wrap_to_array
wrap_from_array = onedal._backend.interop.buffer.wrap_from_array


# TODO: implement more complex logic of
# checking shape & strides in entity
def is_buffer_array(entity) -> bool:
    if is_buffer_entity(entity):
        return is_nd(entity, n=1)
    return False


def to_array(entity):
    assert is_buffer_array(entity)
    return wrap_to_array(entity)


def from_array(array) -> np.ndarray:
    buffer = wrap_from_array(array)
    assert is_buffer_array(buffer)
    return np.asarray(buffer)
