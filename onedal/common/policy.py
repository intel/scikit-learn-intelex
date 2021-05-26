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

try:
    from _onedal4py_dpc import (
        PyPolicy
    )
except ImportError:
    from _onedal4py_host import (
        PyPolicy
    )

def _get_current_policy():
    import sys
    if 'daal4py.oneapi' in sys.modules:
        import daal4py.oneapi as d4p_oneapi
        devname = d4p_oneapi._get_device_name_sycl_ctxt()

        if devname == 'gpu':
            # daal4py.oneapi.sycl_context uses default_selector only to create a queue
            # so we do not have to extract a queue object from daal4py and just create it
            # one more time
            return PyPolicy('gpu')
    return PyPolicy('host')
