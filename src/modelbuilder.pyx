cdef extern from "modelbuilder.h":
    ctypedef size_t c_NodeId
    ctypedef size_t c_TreeId
    cdef size_t c_noParent

    cdef cppclass c_decision_forest_classification_ModelBuilder:
        ModelBuilder(size_t nClasses, size_t nTrees) except +
        c_TreeId createTree(size_t nNodes)
        c_NodeId addLeafNode(c_TreeId treeId, c_NodeId parentId, size_t position, size_t classLabel)
        c_NodeId addSplitNode(c_TreeId treeId, c_NodeId parentId, size_t position, size_t featureIndex, double featureValue)
        ModelPtr getModel()


cdef class decision_forest_classification_modelbuilder:

    cdef c_decision_forest_classification_ModelBuilder * c_ptr

    def __cinit__(self, size_t nClasses, size_t nTrees):
        self.c_ptr = new c_decision_forest_classification_ModelBuilder(nClasses, nTrees)

    def __dealloc__(self):
        del self.c_ptr

    def createTree(self, size_t nNodes):
        return self.c_ptr.createTree(nNodes)

    def addLeafNode(self, c_TreeId treeId, c_NodeId parentId, size_t position, size_t classLabel):
        return self.c_ptr.addLeafNode(treeId, parentId, position, classLabel)

    def addSplitNode(self, c_TreeId treeId, size_t position, size_t featureIndex, double featureValue, c_NodeId parentId=c_noParent):
        return self.c_ptr.addSplitNode(treeId, parentId, position, featureIndex, featureValue)

    def getModel(self):
        return self.c_ptr.getModel()


def model_builder(nTrees=1, nClasses=2):
    '''
    Currently we support only decision forest classification models.
    The future may bring us more.
    '''
    return decision_forest_classification_modelbuilder(nClasses, nTrees)
