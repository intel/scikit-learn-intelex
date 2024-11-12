# ===============================================================================
# Copyright 2024 Intel Corporation
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
# ===============================================================================

from ._dataframes_support import get_dataframes_and_queues
from ._device_selection import (
    get_memory_usm,
    get_queues,
    is_dpctl_device_available,
    pass_if_not_implemented_for_gpu,
)

__all__ = [
    "get_dataframes_and_queues",
    "get_queues",
    "get_memory_usm",
    "is_dpctl_device_available",
    "pass_if_not_implemented_for_gpu",
]
