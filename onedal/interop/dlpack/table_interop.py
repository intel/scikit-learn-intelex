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

from .dlpack_utils import is_dlpack_entity, is_nd

wrap_to_homogen_table = onedal._backend.interop.dlpack.wrap_to_homogen_table


def is_dlpack_table(entity) -> bool:
    if is_dlpack_entity(entity):
        return is_nd(entity, n=2)
    return False


def to_homogen_table(entity):
    assert is_dlpack_table(entity)
    return wrap_to_homogen_table(entity)
