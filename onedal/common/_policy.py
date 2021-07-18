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
    import onedal._onedal_py_dpc as backend
except ImportError:
    import onedal._onedal_py_host as backend


class _HostPolicy(backend.host_policy):

    def __init__(self):
        import sys

        self.host_ctx, self.gpu_ctx = None, None
        if 'daal4py.oneapi' in sys.modules:
            import daal4py.oneapi as d4p_oneapi
            devname = d4p_oneapi._get_device_name_sycl_ctxt()
            if devname == 'gpu':
                self.gpu_ctx = d4p_oneapi._get_sycl_ctxt()
                self.host_ctx = d4p_oneapi.sycl_execution_context('host')
                self.host_ctx.apply()
        super().__init__()

    def __del__(self):
        if self.host_ctx:
            del self.host_ctx
        if self.gpu_ctx:
            self.gpu_ctx.apply()
