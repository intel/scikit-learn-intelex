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

    cdef TreeState _getTreeState[M](M & model, size_t i, size_t n_classes)


cdef class pyTreeState(object):
    cdef skl_tree_node[:] node_ar
    cdef double[:] value_ar
    cdef size_t max_depth
    cdef size_t node_count
    cdef size_t leaf_count
    cdef size_t class_count


    cdef set(self, TreeState * treeState):
        self.max_depth = treeState.max_depth
        self.node_count = treeState.node_count
        self.leaf_count = treeState.leaf_count
        self.class_count = treeState.class_count
        self.node_ar = <skl_tree_node[:self.node_count]>treeState.node_ar
        self.value_ar = <double[:self.node_count]>treeState.value_ar

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
    elif isinstance(model, decision_forest_regression_model):
        cTreeState = getTreeState_df_regression_model(model, i, n_classes)
    else:
        assert(False), 'Incorrect model type: ' + str(type(model))
    #cdef TreeState p = _getTreeState(model.c_ptr, i, n_classes)
    state = pyTreeState()
    state.set(&cTreeState)
    return state


cdef TreeState getTreeState_df_classification_model(decision_forest_classification_model model, i, n_classes):
    return _getTreeState(model.c_ptr, i, n_classes)


cdef TreeState getTreeState_df_regression_model(decision_forest_regression_model model, i, n_classes):
    return _getTreeState(model.c_ptr, i, n_classes)

