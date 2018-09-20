from cpython cimport Py_INCREF, PyTypeObject
import numpy as np
cimport numpy as np


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    object PyArray_SimpleNewFromData(int nd, np.npy_intp* dims, int typenum, void* data)


cdef extern from "tree_visitor.h":
    cdef struct skl_tree_node:
        ssize_t left_child
        ssize_t right_child
        ssize_t feature
        double threshold
        double impurity
        ssize_t n_node_samples
        double weighted_n_node_samples

    cdef struct TreeState:
        skl_tree_node *node_ar
        double        *value_ar
        size_t         max_depth
        size_t         node_count
        size_t         leaf_count
        size_t         class_count

    #cdef TreeState _getTreeState[M](M & model, size_t i, size_t n_classes)
    cdef TreeState _getTreeStateClassification(decision_forest_classification_ModelPtr * model, size_t i, size_t n_classes)
    #cdef TreeState _getTreeStateRegression(decision_forest_regression_ModelPtr * model, size_t i, size_t n_classes)


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
    cdef np.ndarray node_ar
    #cdef double[:] value_ar
    cdef np.ndarray value_ar
    cdef size_t max_depth
    cdef size_t node_count
    cdef size_t leaf_count
    cdef size_t class_count

    cdef np.ndarray _get_node_ndarray(self, void* nodes, size_t count):
        """Wraps nodes as a NumPy struct array.
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(skl_tree_node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject*> np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_value_ndarray(self, void* values, size_t count):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> count
        cdef np.ndarray arr
        arr = PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, values)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr


    cdef set(self, TreeState * treeState):
        self.max_depth = treeState.max_depth
        self.node_count = treeState.node_count
        self.leaf_count = treeState.leaf_count
        self.class_count = treeState.class_count
        #self.value_ar = <double[:self.node_count]>treeState.value_ar
        self.node_ar = self._get_node_ndarray(<void*> treeState.node_ar, treeState.node_count)
        self.value_ar = self._get_value_ndarray(<void*> treeState.value_ar, treeState.node_count)


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


def getTreeState(model, i, n_classes):
    #state = getTreeState_decision_forest_classification_model(model, i, n_classes)
    cdef TreeState cTreeState
    if isinstance(model, decision_forest_classification_model):
        cTreeState = getTreeState_df_classification_model(model, i, n_classes)
#    elif isinstance(model, decision_forest_regression_model):
#        cTreeState = getTreeState_df_regression_model(model, i, n_classes)
    else:
        assert(False), 'Incorrect model type: ' + str(type(model))
    #cdef TreeState p = _getTreeState(model.c_ptr, i, n_classes)
    state = pyTreeState()
    state.set(&cTreeState)
    return state


cdef TreeState getTreeState_df_classification_model(decision_forest_classification_model model, i, n_classes):
    return _getTreeStateClassification(model.c_ptr, i, n_classes)


#cdef TreeState getTreeState_df_regression_model(decision_forest_regression_model model, i, n_classes):
#    return _getTreeState(model.c_ptr, i, n_classes)

