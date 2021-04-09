#===============================================================================
# Copyright 2021 Intel Corporation
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
#===============================================================================


from functools import wraps
from importlib import import_module


def _execute_with_dpc_or_host(*name_classes):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            import sys

            retval = function(*args, **kwargs)
            return retval
        return wrapper
    return decorator
