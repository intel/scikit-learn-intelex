
from libcpp.string cimport string

cdef extern from "data/backend/utils.h":
    cdef string to_std_string(PyObject * o) except +

cdef extern from "oneapi/dal/algo/svm.hpp" namespace "oneapi::dal::svm::task":
    cdef cppclass classification:
        pass
    cdef cppclass regression:
        pass


cdef extern from "oneapi/dal/algo/svm.hpp" namespace "oneapi::dal::svm":
    cdef cppclass model[task_t]:
        pass

cdef extern from "svm/backend/svm_py.h" namespace "oneapi::dal::python":
    ctypedef struct svm_params:
        string method
        string kernel
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

    cdef cppclass svm_train[task_t]:
        svm_train(svm_params *) except +
        void train(PyObject * data, PyObject * labels, PyObject * weights) except +
        int get_support_vector_count()  except +
        PyObject * get_support_vectors() except +
        PyObject * get_support_indices() except +
        PyObject * get_coeffs() except +
        PyObject * get_biases() except +
        model[task_t] get_model() except +

    cdef cppclass svm_infer[task_t]:
        svm_infer(svm_params *) except +
        void infer(PyObject * data, PyObject * support_vectors, PyObject * coeffs, PyObject * biases) except +
        void infer(PyObject * data, model[task_t] * model) except +
        PyObject * get_labels() except +
        PyObject * get_decision_function() except +

