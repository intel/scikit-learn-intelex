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

def get_config():
    """Retrieve current values of global configuration
    Returns
    -------
    config : dict
        Keys are parameter names
    """
    import sys
    if 'daal4py.oneapi' in sys.modules:
        import daal4py.oneapi as d4p_oneapi
        devname = d4p_oneapi._get_device_name_sycl_ctxt()
        params = d4p_oneapi._get_sycl_ctxt_params()

        return {
            'target_offload': 'host' if devname == 'cpu' else devname,
            'allow_fallback_to_host': params.get('host_offload_on_fail', False)
        }
    return {
        'target_offload': None,
        'allow_fallback_to_host': False
    }
