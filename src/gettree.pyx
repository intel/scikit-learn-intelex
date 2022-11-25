#===============================================================================
# Copyright 2020 Intel Corporation
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

from cpython cimport Py_INCREF, PyTypeObject
import numpy as np
cimport numpy as cnp


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    object PyArray_SimpleNewFromData(int nd, cnp.npy_intp* dims, int typenum, void* data)

cdef extern from "daal4py.h":
    cdef void set_rawp_base(cnp.ndarray, void *)

cdef extern from "tree_visitor.h":
    cdef struct skl_tree_node:
        Py_ssize_t left_child
        Py_ssize_t right_child
        Py_ssize_t feature
        double threshold
        double impurity
        Py_ssize_t n_node_samples
        double weighted_n_node_samples

    cdef struct TreeState:
        skl_tree_node *node_ar
        double        *value_ar
        size_t         max_depth
        size_t         node_count
        size_t         leaf_count
        size_t         class_count

    cdef TreeState _getTreeState[M](M * model, size_t i, size_t n_classes)
    cdef TreeState _getTreeState[M](M * model, size_t n_classes)

NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<skl_tree_node*> NULL).left_child,
        <Py_ssize_t> &(<skl_tree_node*> NULL).right_child,
        <Py_ssize_t> &(<skl_tree_node*> NULL).feature,
        <Py_ssize_t> &(<skl_tree_node*> NULL).threshold,
        <Py_ssize_t> &(<skl_tree_node*> NULL).impurity,
        <Py_ssize_t> &(<skl_tree_node*> NULL).n_node_samples,
        <Py_ssize_t> &(<skl_tree_node*> NULL).weighted_n_node_samples
    ]
})


cdef class pyTreeState(object):
    cdef cnp.ndarray node_ar
    cdef cnp.ndarray value_ar
    cdef size_t max_depth
    cdef size_t node_count
    cdef size_t leaf_count
    cdef size_t class_count

    cdef cnp.ndarray _get_node_ndarray(self, void* nodes, size_t count):
        """Wraps nodes as a NumPy struct array.
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef cnp.npy_intp shape[1]
        shape[0] = <cnp.npy_intp> count
        cdef cnp.npy_intp strides[1]
        strides[0] = sizeof(skl_tree_node)
        cdef cnp.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject*> cnp.ndarray, <cnp.dtype> NODE_DTYPE, 1, shape,
                                   strides, nodes,
                                   cnp.NPY_DEFAULT, None)
        set_rawp_base(arr, nodes)
        return arr


    cdef cnp.ndarray _get_value_ndarray(self, void* values, size_t count, size_t outputs, size_t class_counts):
        cdef cnp.npy_intp shape[3]
        shape[0] = <cnp.npy_intp> count
        shape[1] = <cnp.npy_intp> 1
        shape[2] = <cnp.npy_intp> class_counts
        cdef cnp.ndarray arr
        arr = PyArray_SimpleNewFromData(3, shape, cnp.NPY_DOUBLE, values)
        set_rawp_base(arr, values)
        return arr


    cdef set(self, TreeState * treeState):
        self.max_depth = treeState.max_depth
        self.node_count = treeState.node_count
        self.leaf_count = treeState.leaf_count
        self.class_count = treeState.class_count
        self.node_ar = self._get_node_ndarray(<void*> treeState.node_ar, treeState.node_count)
        self.value_ar = self._get_value_ndarray(<void*> treeState.value_ar, treeState.node_count, 1, treeState.class_count)


    @property
    def node_ar(self):
        return self.node_ar

    @property
    def value_ar(self):
        return self.value_ar

    @property
    def max_depth(self):
        return self.max_depth

    @property
    def node_count(self):
        return self.node_count

    @property
    def leaf_count(self):
        return self.leaf_count

    @property
    def class_count(self):
        return self.class_count
