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

cdef extern from "common/backend/utils.h" namespace "oneapi::dal::python":
    cdef std_string to_std_string(PyObject * o) except +

cdef extern from "oneapi/dal/algo/svm.hpp" namespace "oneapi::dal::svm::task":
    cdef cppclass classification:
        pass
    cdef cppclass regression:
        pass

cdef extern from "svm/backend/svm_py.h" namespace "oneapi::dal::python":
    cdef cppclass svm_params:
        std_string method
        std_string kernel
        int class_count
        double c
        double epsilon
        double accuracy_threshold
        int max_iteration_count
        double tau
        double cache_size
        bool shrinking
        double shift
        double scale
        int degree
        double sigma

    cdef cppclass svm_model[task_t]:
        svm_model() except +
        object serialize() except +
        void deserialize(object bytes) except +

    cdef cppclass svm_train[task_t]:
        svm_train(svm_params *) except +
        void train(PyObject * data, PyObject * labels, PyObject * weights) except +
        int get_support_vector_count()  except +
        PyObject * get_support_vectors() except +
        PyObject * get_support_indices() except +
        PyObject * get_coeffs() except +
        PyObject * get_biases() except +
        svm_model[task_t] get_model() except +

    cdef cppclass svm_infer[task_t]:
        svm_infer(svm_params *) except +
        void infer(PyObject * data, PyObject * support_vectors, PyObject * coeffs, PyObject * biases) except +
        void infer(PyObject * data, svm_model[task_t] * model) except +
        PyObject * get_labels() except +
        PyObject * get_decision_function() except +
