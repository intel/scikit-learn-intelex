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

from .sua_utils import convert_sua, is_nd, is_sua_entity

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
