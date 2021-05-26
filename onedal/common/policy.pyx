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

include "policy.pxi"

cdef class PyPolicy:
    cdef policy* _policy

    def __cinit__(self, type_):
        self._policy = new policy(to_std_string( <PyObject*>type_))

    def __dealloc__(self):
        del self._policy

    cdef const policy* get_cref(self):
        return self._policy
