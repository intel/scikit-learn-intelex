#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
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
#******************************************************************************/

# We expose DAAL's directly through Cython
# Currently only GBT is available.
# The model builder object is retrieved through calling model_builder.
# We will extend this once we know how other model builders willl work in DAAL

cdef extern from "modelbuilder.h":
    ctypedef size_t c_gbt_clf_NodeId
    ctypedef size_t c_gbt_clf_TreeId
    ctypedef size_t c_gbt_reg_NodeId
    ctypedef size_t c_gbt_reg_TreeId

    cdef size_t c_gbt_clf_noParent
    cdef size_t c_gbt_reg_noParent

    cdef cppclass c_gbt_classification_ModelBuilder:
        c_gbt_classification_ModelBuilder(size_t nFeatures, size_t nIterations, size_t nClasses) except +
        c_gbt_clf_TreeId createTree(size_t nNodes, size_t classLabel)
        c_gbt_clf_NodeId addLeafNode(c_gbt_clf_TreeId treeId, c_gbt_clf_NodeId parentId, size_t position, double response)
        c_gbt_clf_NodeId addSplitNode(c_gbt_clf_TreeId treeId, c_gbt_clf_NodeId parentId, size_t position, size_t featureIndex, double featureValue)

    cdef cppclass c_gbt_regression_ModelBuilder:
        c_gbt_regression_ModelBuilder(size_t nFeatures, size_t nIterations) except +
        c_gbt_reg_TreeId createTree(size_t nNodes)
        c_gbt_reg_NodeId addLeafNode(c_gbt_reg_TreeId treeId, c_gbt_reg_NodeId parentId, size_t position, double response)
        c_gbt_reg_NodeId addSplitNode(c_gbt_reg_TreeId treeId, c_gbt_reg_NodeId parentId, size_t position, size_t featureIndex, double featureValue)

    cdef gbt_classification_ModelPtr * get_gbt_classification_modelbuilder_Model(c_gbt_classification_ModelBuilder *)
    cdef gbt_regression_ModelPtr * get_gbt_regression_modelbuilder_Model(c_gbt_regression_ModelBuilder *)


cdef class gbt_classification_modelbuilder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_classification_ModelBuilder * c_ptr

    def __cinit__(self, size_t n_features, size_t n_iterations, size_t n_classes):
        self.c_ptr = new c_gbt_classification_ModelBuilder(n_features, n_iterations, n_classes)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes, size_t class_label):
        '''
        Create certain tree in the gradient boosted trees classification model for certain class

        :param size_t n_nodes: number of nodes in created tree
        :param size_t class_label: label of class for which tree is created. classLabel bellows interval from 0 to (nClasses - 1)
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes, class_label)

    def add_leaf(self, c_gbt_clf_TreeId tree_id, double response, c_gbt_clf_NodeId parent_id=c_gbt_clf_noParent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle tree_id: tree to which new node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :rtype: node identifier
        '''
        return self.c_ptr.addLeafNode(tree_id, parent_id, position, response)

    def add_split(self, c_gbt_clf_TreeId tree_id, size_t feature_index, double feature_value, c_gbt_clf_NodeId parent_id=c_gbt_clf_noParent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle tree_id: tree to which node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t feature_index: feature index for spliting
        :param double feature_value: feature value for spliting
        :rtype: node identifier
        '''
        return self.c_ptr.addSplitNode(tree_id, parent_id, position, feature_index, feature_value)

    def model(self):
        '''
        Get built model

        :rtype: gbt_classification_model
        '''
        cdef gbt_classification_model res = gbt_classification_model.__new__(gbt_classification_model)
        res.c_ptr = get_gbt_classification_modelbuilder_Model(self.c_ptr)
        return res


cdef class gbt_regression_modelbuilder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_regression_ModelBuilder * c_ptr

    def __cinit__(self, size_t n_features, size_t n_iterations):
        self.c_ptr = new c_gbt_regression_ModelBuilder(n_features, n_iterations)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes):
        '''
        Create certain tree in the gradient boosted trees regression model

        :param size_t n_nodes: number of nodes in created tree
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes)

    def add_leaf(self, c_gbt_reg_TreeId tree_id, double response, c_gbt_reg_NodeId parent_id=c_gbt_reg_noParent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle tree_id: tree to which new node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :rtype: node identifier
        '''
        return self.c_ptr.addLeafNode(tree_id, parent_id, position, response)

    def add_split(self, c_gbt_reg_TreeId tree_id, size_t feature_index, double feature_value, c_gbt_reg_NodeId parent_id=c_gbt_reg_noParent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle tree_id: tree to which node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t feature_index: feature index for spliting
        :param double feature_value: feature value for spliting
        :rtype: node identifier
        '''
        return self.c_ptr.addSplitNode(tree_id, parent_id, position, feature_index, feature_value)

    def model(self):
        '''
        Get built model

        :rtype: gbt_regression_model
        '''
        cdef gbt_regression_model res = gbt_regression_model.__new__(gbt_regression_model)
        res.c_ptr = get_gbt_regression_modelbuilder_Model(self.c_ptr)
        return res


def gbt_clf_model_builder(n_features, n_iterations, n_classes = 2):
    '''
    Model builder for gradient boosted trees classification

    :param size_t n_features: Number of features in training data
    :param size_t n_iterations: Number of trees in model for each class
    :param size_t n_classes: Number of classes in model
    '''
    return gbt_classification_modelbuilder(n_features, n_iterations, n_classes)


def gbt_reg_model_builder(n_features, n_iterations):
    '''
    Model builder for gradient boosted trees regression

    :param size_t n_features: Number of features in training data
    :param size_t n_iterations: Number of trees in the model
    '''
    return gbt_regression_modelbuilder(n_features, n_iterations)
