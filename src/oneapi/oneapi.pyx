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
from cpython.ref cimport PyObject, Py_INCREF
from libcpp cimport bool


cdef extern from "oneapi/oneapi.h":
    cdef cppclass PySyclExecutionContext:
        PySyclExecutionContext(const std_string & dev) except +
    void * tosycl(void *, int, int*)
    void * todaalnt(void*, int, int*)
    void del_scl_buffer(void *, int)

    std_string to_std_string(PyObject * o) except +

    void * c_make_py_from_sycltable(void * ptr, int typ) except +
    void c_set_queue_to_daal_context(PyObject* queue_object) except +
    void c_reset_daal_context() except +



cdef class sycl_execution_context:
    cdef PySyclExecutionContext * c_ptr

    def __cinit__(self, dev):
        self.c_ptr = new PySyclExecutionContext(to_std_string(<PyObject *>dev))

    def __dealloc__(self):
        del self.c_ptr


# thread-local storage
from threading import local as threading_local
_tls = threading_local()

def _is_tls_initialized():
    return (getattr(_tls, 'initialized', None) is not None) and (_tls.initialized == True)

def _initialize_tls():
    _tls._in_sycl_ctxt = False
    _tls.initialized = True

def _set_in_sycl_ctxt(value):
    if not _is_tls_initialized():
        _initialize_tls()
    _tls._in_sycl_ctxt = value

def _get_in_sycl_ctxt():
    if not _is_tls_initialized():
        _initialize_tls()
    return _tls._in_sycl_ctxt

def is_in_sycl_ctxt():
    return _get_in_sycl_ctxt()


from contextlib import contextmanager

@contextmanager
def sycl_context(dev='default'):
    # Code to acquire resource
    ctxt = sycl_execution_context(dev)
    _set_in_sycl_ctxt(True)
    try:
        yield ctxt
    finally:
        # Code to release resource
        del ctxt
        _set_in_sycl_ctxt(False)


cimport numpy as np
import numpy as np
from cpython.pycapsule cimport PyCapsule_New

cdef class sycl_buffer:
    'Sycl buffer for DAAL. A generic implementation needs to do much more.'

    cdef readonly long sycl_buffer
    cdef int typ
    cdef int shape[2]
    cdef object _ary


    def __cinit__(self, ary=None):
        self._ary = ary
        if ary is not None:
            assert ary.flags['C_CONTIGUOUS'] and ary.ndim == 2
            self.__inilz__(0, np.PyArray_TYPE(ary), ary.shape[0], ary.shape[1])

    cpdef __inilz__(self, long b, int t, int d1, int d2):
        self.typ = t
        self.shape[0] = d1
        self.shape[1] = d2
        if b:
            self.sycl_buffer = b
        else:
            self.sycl_buffer = <long>tosycl(np.PyArray_DATA(self._ary), self.typ, self.shape)

    def __dealloc__(self):
        del_scl_buffer(<void*>self.sycl_buffer, self.typ)

    # we need to consider how to make this usable by numba/HPAT without objmode
    def __2daalnt__(self):
        return PyCapsule_New(todaalnt(<void*>self.sycl_buffer, self.typ, self.shape), NULL, NULL)

cdef api object make_py_from_sycltable(void * ptr, int typ, int d1, int d2):
    if not _get_in_sycl_ctxt():
        return None
    cdef void * buff = c_make_py_from_sycltable(ptr, typ)
    if buff:
        res = sycl_buffer.__new__(sycl_buffer)
        res.__inilz__(<long>buff, typ, d1, d2)
        return res
    return None


def set_queue_to_daal_context(queue):
    c_set_queue_to_daal_context(<PyObject*>queue)

def reset_daal_context():
    c_reset_daal_context()
