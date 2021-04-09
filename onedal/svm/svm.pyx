
# distutils: language = c++
# cython: language_level=2

# Import the Python-level symbols of numpy

# Import the C-level symbols of numpy
import cython
from libcpp cimport bool
from cpython.ref cimport PyObject
cimport numpy as npc


include "svm.pxi"


@cython.auto_pickle(True)
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
    cdef model[classification] * thisptr

    def __cinit__(self):
        self.thisptr = new model[classification]()

    def __dealloc__(self):
        del self.thisptr

    def __setstate__(self, state):
        # TODO: need serialize of models on c++ side
        # if isinstance(state, bytes):
        #    self.thisptr = svm_model[svm_model[classification]](state)
        # else:
        raise NotImplementedError("serialize not avalible for onedal models")

    def __getstate__(self):
        raise NotImplementedError("serialize not avalible for onedal models")


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

# IF ONEDAL_VERSION >= ONEDAL_2021_3_VERSION:

cdef class PyRegressionSvmModel:
    cdef model[regression] * thisptr

    def __cinit__(self):
        self.thisptr = new model[regression]()

    def __dealloc__(self):
        del self.thisptr

    def __setstate__(self, state):
        # TODO: need serialize of models on c++ side
        # if isinstance(state, bytes):
        #    self.thisptr = svm_model[svm_model[regression]](state)
        # else:
        raise NotImplementedError("serialize not avalible for onedal models")

    def __getstate__(self):
        raise NotImplementedError("serialize not avalible for onedal models")


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
