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
