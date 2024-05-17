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

host_policy = onedal._backend.host_policy
default_host_policy = onedal._backend.default_host_policy


def check_attr(obj, name: str, checker=None) -> bool:
    if checker is None:
        check = lambda x: True
    else:
        check = checker
    try:
        if hasattr(obj, name):
            attr = getattr(obj, name)
        else:
            attr = obj[name]
        return check(attr)
    except Exception:
        return False


def is_host_policy(policy):
    is_host = isinstance(policy, host_policy)
    is_default = isinstance(policy, default_host_policy)
    return is_host or is_default


def is_cpu_policy(policy):
    is_host = is_host_policy(policy)
    is_cpu = policy.get_device_name() == "cpu"
    return is_host or is_cpu
