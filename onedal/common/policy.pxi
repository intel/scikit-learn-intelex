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

from libcpp.string cimport string as std_string

cdef extern from "common/backend/policy.h" namespace "oneapi::dal::python":
    cdef cppclass policy:
        policy(const std_string& device_name) except +

cdef extern from "common/backend/utils.h" namespace "oneapi::dal::python":
    cdef std_string to_std_string(PyObject * o) except +
