

cdef extern from "prims/backend/kernel_functions_py.h" namespace "oneapi::dal::python":
    ctypedef struct linear_kernel_params:
        double scale
        double shift

    cdef cppclass linear_kernel_compute:
        linear_kernel_compute(linear_kernel_params *) except +
        void compute(PyObject * x, PyObject * y) except +
        PyObject * get_values() except +

    ctypedef struct rbf_kernel_params:
        double sigma

    cdef cppclass rbf_kernel_compute:
        rbf_kernel_compute(rbf_kernel_params *) except +
        void compute(PyObject * x, PyObject * y) except +
        PyObject * get_values() except +

    ctypedef struct polynomial_kernel_params:
        double scale
        double shift
        double degree

    cdef cppclass polynomial_kernel_compute:
        polynomial_kernel_compute(polynomial_kernel_params *) except +
        void compute(PyObject * x, PyObject * y) except +
        PyObject * get_values() except +
