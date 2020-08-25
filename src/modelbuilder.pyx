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
    ctypedef size_t c_NodeId
    ctypedef size_t c_TreeId
    cdef size_t c_noParent

    cdef cppclass c_gbt_classification_ModelBuilder:
        c_gbt_classification_ModelBuilder(size_t nFeatures, size_t nIterations, size_t nClasses) except +
        c_TreeId createTree(size_t nNodes, size_t classLabel)
        c_NodeId addLeafNode(c_TreeId treeId, c_NodeId parentId, size_t position, double response)
        c_NodeId addSplitNode(c_TreeId treeId, c_NodeId parentId, size_t position, size_t featureIndex, double featureValue)

    cdef gbt_classification_ModelPtr * get_gbt_classification_modelbuilder_Model(c_gbt_classification_ModelBuilder *)


cdef class gbt_classification_modelbuilder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_classification_ModelBuilder * c_ptr
    cdef size_t nFeatures

    def __cinit__(self, size_t nFeatures, size_t nIterations, size_t nClasses):
        self.c_ptr = new c_gbt_classification_ModelBuilder(nFeatures, nIterations, nClasses)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t nNodes, size_t classLabel):
        '''
        Create certain tree in the gradient boosted trees classification model for certain class

        :param size_t nNodes: number of nodes in created tree
        :param size_t classLabel: label of class for which tree is created. classLabel bellows interval from 0 to (nClasses - 1)
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(nNodes, classLabel)

    def add_leaf(self, c_TreeId treeId, double response, c_NodeId parentId=c_noParent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle treeId: tree to which new node is added
        :param node-handle parentId: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :rtype: node identifier
        '''
        return self.c_ptr.addLeafNode(treeId, parentId, position, response)

    def add_split(self, c_TreeId treeId, size_t featureIndex, double featureValue, c_NodeId parentId=c_noParent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle treeId: tree to which node is added
        :param node-handle parentId: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t featureIndex: feature index for spliting
        :param double featureValue: feature value for spliting
        :rtype: node identifier
        '''
        return self.c_ptr.addSplitNode(treeId, parentId, position, featureIndex, featureValue)

    def model(self):
        '''
        Get built model

        :rtype: gbt_classification_model
        '''
        cdef gbt_classification_model res = gbt_classification_model.__new__(gbt_classification_model)
        res.c_ptr = get_gbt_classification_modelbuilder_Model(self.c_ptr)
        return res


def model_builder(nFeatures, nIterations, nClasses = 2):
    '''
    Currently we support only gradient bossted trees classification models.
    The future may bring us more.

    :param size_t nFeatures: Number of features in training data
    :param size_t nIterations: Number of trees in model for each class
    :param size_t nClasses: Number of classes in model
    '''
    return gbt_classification_modelbuilder(nFeatures, nIterations, nClasses)
