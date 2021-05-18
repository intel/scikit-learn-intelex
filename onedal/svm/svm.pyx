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

import cython
from libcpp cimport bool
from cpython.ref cimport PyObject
cimport numpy as npc


include "svm.pxi"

cdef class PySvmParams:
    cdef svm_params pt

    def __init__(self, method, kernel, class_count, c,
                 epsilon, accuracy_threshold, max_iteration_count,
                 tau, sigma, shift, scale, degree):
        self.pt.method = to_std_string( < PyObject * >method)
        self.pt.kernel = to_std_string( < PyObject * >kernel)
        self.pt.class_count = class_count
        self.pt.c = c
        self.pt.epsilon = epsilon
        self.pt.accuracy_threshold = accuracy_threshold
        self.pt.max_iteration_count = max_iteration_count
        self.pt.tau = tau
        self.pt.sigma = sigma
        self.pt.shift = shift
        self.pt.scale = scale
        self.pt.degree = degree


cdef class PyClassificationSvmModel:
    cdef svm_model[classification] * thisptr

    def __cinit__(self):
        self.thisptr = new svm_model[classification]()

    def __dealloc__(self):
        del self.thisptr

    def __setstate__(self, state):
        if isinstance(state, bytes):
           self.thisptr.deserialize(state)
        else:
           raise ValueError("Invalid state")

    def __getstate__(self):
        if self.thisptr == NULL:
            raise ValueError("Pointer to svm model is NULL")
        bytes = self.thisptr.serialize()
        return bytes


cdef class PyClassificationSvmTrain:
    cdef svm_train[classification] * thisptr

    def __cinit__(self, PySvmParams params):
        self.thisptr = new svm_train[classification](& params.pt)

    def __dealloc__(self):
        del self.thisptr

    def train(self, data, labels, weights):
        self.thisptr.train( < PyObject * >data, < PyObject * >labels, < PyObject * >weights)

    def get_support_vectors(self):
        return < object > self.thisptr.get_support_vectors()

    def get_support_indices(self):
        return < object > self.thisptr.get_support_indices()

    def get_coeffs(self):
        return < object > self.thisptr.get_coeffs()

    def get_biases(self):
        return < object > self.thisptr.get_biases()

    def get_model(self):
        cdef PyClassificationSvmModel res = PyClassificationSvmModel.__new__(PyClassificationSvmModel)
        res.thisptr[0] = self.thisptr.get_model()
        return res


cdef class PyClassificationSvmInfer:
    cdef svm_infer[classification] * thisptr

    def __cinit__(self, PySvmParams params):
        self.thisptr = new svm_infer[classification](& params.pt)

    def __dealloc__(self):
        del self.thisptr

    def infer(self, data, PyClassificationSvmModel model):
        self.thisptr.infer( < PyObject * >data, model.thisptr)

    def infer_builder(self, data, support_vectors, coeffs, biases):
        self.thisptr.infer( < PyObject * >data, < PyObject * >support_vectors, < PyObject * >coeffs, < PyObject * >biases)

    def get_labels(self):
        return < object > self.thisptr.get_labels()

    def get_decision_function(self):
        return < object > self.thisptr.get_decision_function()


cdef class PyRegressionSvmModel:
    cdef svm_model[regression] * thisptr

    def __cinit__(self):
        self.thisptr = new svm_model[regression]()

    def __dealloc__(self):
        del self.thisptr

    def __setstate__(self, state):
        if isinstance(state, bytes):
           self.thisptr.deserialize(state)
        else:
           raise ValueError("Invalid state")

    def __getstate__(self):
        if self.thisptr == NULL:
            raise ValueError("Pointer to svm model is NULL")
        bytes = self.thisptr.serialize()
        return bytes


cdef class PyRegressionSvmTrain:
    cdef svm_train[regression] * thisptr

    def __cinit__(self, PySvmParams params):
        self.thisptr = new svm_train[regression](& params.pt)

    def __dealloc__(self):
        del self.thisptr

    def train(self, data, labels, weights):
        self.thisptr.train( < PyObject * >data, < PyObject * >labels, < PyObject * >weights)

    def get_support_vectors(self):
        return < object > self.thisptr.get_support_vectors()

    def get_support_indices(self):
        return < object > self.thisptr.get_support_indices()

    def get_coeffs(self):
        return < object > self.thisptr.get_coeffs()

    def get_biases(self):
        return < object > self.thisptr.get_biases()

    def get_model(self):
        cdef PyRegressionSvmModel res = PyRegressionSvmModel.__new__(PyRegressionSvmModel)
        res.thisptr[0] = self.thisptr.get_model()
        return res


cdef class PyRegressionSvmInfer:
    cdef svm_infer[regression] * thisptr

    def __cinit__(self, PySvmParams params):
        self.thisptr = new svm_infer[regression](& params.pt)

    def __dealloc__(self):
        del self.thisptr

    def infer(self, data, PyRegressionSvmModel model):
        self.thisptr.infer( < PyObject * >data, model.thisptr)

    def infer_builder(self, data, support_vectors, coeffs, biases):
        self.thisptr.infer( < PyObject * >data, < PyObject * >support_vectors, < PyObject * >coeffs, < PyObject * >biases)

    def get_labels(self):
        return < object > self.thisptr.get_labels()
