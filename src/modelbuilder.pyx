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
# Currently only DF is available.
# The model builder object is retrieved through calling model_builder.
# We will extend this once we know how other model builders willl work in DAAL

cdef extern from "modelbuilder.h":
    ctypedef size_t c_NodeId
    ctypedef size_t c_TreeId
    cdef size_t c_noParent

    cdef cppclass c_decision_forest_classification_ModelBuilder:
        c_decision_forest_classification_ModelBuilder(size_t nClasses, size_t nTrees) except +
        c_TreeId createTree(size_t nNodes)
        c_NodeId addLeafNode(c_TreeId treeId, c_NodeId parentId, size_t position, size_t classLabel)
        c_NodeId addSplitNode(c_TreeId treeId, c_NodeId parentId, size_t position, size_t featureIndex, double featureValue)

    cdef decision_forest_classification_ModelPtr * get_decision_forest_classification_modelbuilder_Model(c_decision_forest_classification_ModelBuilder *);


cdef class decision_forest_classification_modelbuilder:
    '''
    Model Builder for decision forest.
    '''
    cdef c_decision_forest_classification_ModelBuilder * c_ptr

    def __cinit__(self, size_t nClasses, size_t nTrees):
        self.c_ptr = new c_decision_forest_classification_ModelBuilder(nClasses, nTrees)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t nNodes):
        '''
        Create one tree of the forest.

        :param size_t nNodes: Number of nodes this tree will have
        :rtype: tree-handle
        '''
        return self.c_ptr.createTree(nNodes)

    def add_leaf(self, c_TreeId treeId, size_t classLabel, c_NodeId parentId=c_noParent, size_t position=0):
        '''
        Add a leaf node to given tree.

        :param tree-handle treeId: tree to which node should be added
        :param size_t classLabel: class-label
        :param node-handle parentId: parent of new node, if not given add a root node [optional, default: no parent] 
        :param size_t position: position in root node [optional, default: 0]
        :rtype: node-handle
        '''
        return self.c_ptr.addLeafNode(treeId, parentId, position, classLabel)

    def add_split(self, c_TreeId treeId, size_t featureIndex, double featureValue, c_NodeId parentId=c_noParent, size_t position=0):
        '''
        Add a split node to given tree.

        :param tree-handle treeId: tree to which node should be added
        :param size_t featureIndex: index of split feature
        :param double featureValue: split value
        :param node-handle parentId: parent of new node, if not given add a root node [optional, default: no parent] 
        :param size_t position: position in root node [optional, default: 0]
        :rtype: node-handle
        '''
        return self.c_ptr.addSplitNode(treeId, parentId, position, featureIndex, featureValue)

    def model(self):
        '''
        Return the decision forest model to be used for decision forest prediction

        :rtype: decision_forest_classification_model
        '''
        cdef decision_forest_classification_model res = decision_forest_classification_model.__new__(decision_forest_classification_model)
        res.c_ptr = get_decision_forest_classification_modelbuilder_Model(self.c_ptr)
        return res


def model_builder(nTrees=1, nClasses=2):
    '''
    Currently we support only decision forest classification models.
    The future may bring us more.

    :param size_t nTrees: Number of trees in the decision forest
    :param size_t nClasses: Number of classes in model
    '''
    return decision_forest_classification_modelbuilder(nClasses, nTrees)
