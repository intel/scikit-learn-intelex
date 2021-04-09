
# distutils: language = c++
#cython: language_level=2

# Import the Python-level symbols of numpy

# Import the C-level symbols of numpy
cimport numpy as npc

npc.import_array()

include "svm/svm.pyx"
include "prims/kernel_functions.pyx"
