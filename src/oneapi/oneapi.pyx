#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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
#******************************************************************************/

# distutils: language = c++
# cython: language_level=2

from libcpp.string cimport string as std_string
from cpython.ref cimport PyObject


cdef extern from "oneapi/oneapi.h":
    cdef cppclass PySyclExecutionContext:
        PySyclExecutionContext(const std_string & dev) except +
    void * tosycl(void *, int, int*)
    void * todaalnt(void*, int, int*)
    void del_scl_buffer(void *, int)

    cdef std_string to_std_string(PyObject * o) except +


cdef class sycl_execution_context:
    cdef PySyclExecutionContext * c_ptr

    def __cinit__(self, dev):
        self.c_ptr = new PySyclExecutionContext(to_std_string(<PyObject *>dev))

    def __dealloc__(self):
        del self.c_ptr


from contextlib import contextmanager


@contextmanager
def sycl_context(dev='default'):
    # Code to acquire resource
    ctxt = sycl_execution_context(dev)

    try:
        yield ctxt
    finally:
        # Code to release resource
        del ctxt


cimport numpy as np
import numpy as np
from cpython.pycapsule cimport PyCapsule_New

cdef class sycl_buffer:
    'Sycl buffer for DAAL. A generic implementation needs to do much more.'

    cdef void * sycl_buffer
    cdef int typ
    cdef int shape[2]
    cdef object _ary

    def __cinit__(self, ary):
        print(type(ary))
        assert ary.flags['C_CONTIGUOUS'] and ary.ndim == 2
        self._ary = ary
        self.typ = np.PyArray_TYPE(ary)
        self.shape[0] = ary.shape[0]
        self.shape[1] = ary.shape[1]
        print(self.typ, self.shape)
        self.sycl_buffer = tosycl(np.PyArray_DATA(ary), self.typ, self.shape)

    def __dealloc__(self):
        del_scl_buffer(self.sycl_buffer, self.typ)

    # we need to consider how to make this usable by numba/HPAT without objmode
    @property
    def __2daalnt__(self):
        return PyCapsule_New(todaalnt(self.sycl_buffer, self.typ, self.shape), NULL, NULL)
