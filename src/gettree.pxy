
cdef extern from "tree_visitor.h":
    cdef public struct skl_tree_node:
        ssize_t left_child
        ssize_t right_child
        ssize_t feature
        double threshold
        double impurity
        ssize_t n_node_samples
        double weighted_n_node_samples

    cdef public struct TreeState:
        skl_tree_node *node_ar
        double        *value_ar
        size_t         max_depth
        size_t         node_count
        size_t         leaf_count
        size_t         class_count

    cdef TreeState _getTreeState[M](M & model, size_t i, size_t n_classes)

def getTreeState(gbt_classification_model model, i, n_classes):
    return _getTreeState(model.c_ptr, i, n_classes)
