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

cdef extern from "prims/backend/kernel_functions_py.h" namespace "oneapi::dal::python":
    cdef cppclass linear_kernel_params:
        double scale
        double shift

    cdef cppclass linear_kernel_compute:
        linear_kernel_compute(linear_kernel_params *) except +
        void compute(PyObject * x, PyObject * y) except +
        PyObject * get_values() except +

    cdef cppclass rbf_kernel_params:
        double sigma

    cdef cppclass rbf_kernel_compute:
        rbf_kernel_compute(rbf_kernel_params *) except +
        void compute(PyObject * x, PyObject * y) except +
        PyObject * get_values() except +

    cdef cppclass polynomial_kernel_params:
        double scale
        double shift
        double degree

    cdef cppclass polynomial_kernel_compute:
        polynomial_kernel_compute(polynomial_kernel_params *) except +
        void compute(PyObject * x, PyObject * y) except +
        PyObject * get_values() except +
