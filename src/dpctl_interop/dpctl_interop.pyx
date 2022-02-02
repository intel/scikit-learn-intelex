#===============================================================================
# Copyright 2014 Intel Corporation
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

# distutils: language = c++
# cython: language_level=2

cdef extern from "dpctl_interop/daal_context_service.h":
    void _dppl_set_current_queue_to_daal_context() except +
    void _dppl_reset_daal_context() except +

def set_current_queue_to_daal_context():
    _dppl_set_current_queue_to_daal_context()

def reset_daal_context():
    _dppl_reset_daal_context()
