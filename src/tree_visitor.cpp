/*******************************************************************************
* Copyright 2014-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#define NO_IMPORT_ARRAY
#include <tree_visitor.h>

template<typename TreeNodeVisitor, typename SplitNodeDescriptor, typename LeafNodeDescriptor>
NodeDepthCountNodeVisitor<TreeNodeVisitor, SplitNodeDescriptor, LeafNodeDescriptor>::NodeDepthCountNodeVisitor()
    : n_nodes(0),
      depth(0),
      n_leaf_nodes(0)
{}

// TODO: Needs to store leaf-node response, and split-node impurity/sample_counts values
template<typename TreeNodeVisitor, typename SplitNodeDescriptor, typename LeafNodeDescriptor>
bool NodeDepthCountNodeVisitor<TreeNodeVisitor, SplitNodeDescriptor, LeafNodeDescriptor>::onLeafNode(const LeafNodeDescriptor &desc)
{
    ++n_nodes;
    ++n_leaf_nodes;
    depth = std::max((const size_t)depth, desc.level);
    return true;
}

template<typename TreeNodeVisitor, typename SplitNodeDescriptor, typename LeafNodeDescriptor>
bool NodeDepthCountNodeVisitor<TreeNodeVisitor, SplitNodeDescriptor, LeafNodeDescriptor>::onSplitNode(const SplitNodeDescriptor &desc)
{
    ++n_nodes;
    depth = std::max((const size_t)depth, desc.level);
    return true;
}


toSKLearnClassificationTreeObjectVisitor::toSKLearnClassificationTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes)
    : node_id(0),
      parents(arange<ssize_t>(-1, _depth-1))
{
    node_count = _n_nodes;
    max_depth = _depth;
    leaf_count = _n_leafs;
    class_count = _max_n_classes;
    node_ar = new skl_tree_node[node_count];
    value_ar = new double[node_count*1*class_count](); // DAAL only supports scalar responses for now
}


bool toSKLearnClassificationTreeObjectVisitor::onSplitNode(const daal::algorithms::tree_utils::classification::SplitNodeDescriptor &desc)
{
    if(desc.level > 0) {
        // has parents
        ssize_t parent = parents[desc.level - 1];
        if(node_ar[parent].left_child > 0) {
            assert(node_ar[node_id].right_child < 0);
            node_ar[parent].right_child = node_id;
        } else {
            node_ar[parent].left_child = node_id;
        }
    }
    
    parents[desc.level] = node_id;
    node_ar[node_id].feature = desc.featureIndex;
    node_ar[node_id].threshold = desc.featureValue;
    node_ar[node_id].impurity = desc.impurity;
    node_ar[node_id].n_node_samples = desc.nNodeSampleCount;
    node_ar[node_id].weighted_n_node_samples = desc.nNodeSampleCount;
    
    // wrap-up
    ++node_id;
    return true;
}

bool toSKLearnClassificationTreeObjectVisitor::onLeafNode(const daal::algorithms::tree_utils::classification::LeafNodeDescriptor &desc)
{
    assert(desc.level > 0);
    if(desc.level) {
        ssize_t parent = parents[desc.level - 1];
        if(node_ar[parent].left_child > 0) {
            assert(node_ar[node_id].right_child < 0);
            node_ar[parent].right_child = node_id;
        } else {
            node_ar[parent].left_child = node_id;
        }
    }
    
    node_ar[node_id].impurity = desc.impurity;
    node_ar[node_id].n_node_samples = desc.nNodeSampleCount;
    node_ar[node_id].weighted_n_node_samples = desc.nNodeSampleCount;
    
    value_ar[node_id*1*class_count + desc.label] += desc.nNodeSampleCount;
    
    // wrap-up
    ++node_id;
    return true;
}

toSKLearnRegressionTreeObjectVisitor::toSKLearnRegressionTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes)
    : node_id(0),
      parents(arange<ssize_t>(-1, _depth-1))
{
    node_count = _n_nodes;
    max_depth = _depth;
    leaf_count = _n_leafs;
    class_count = _max_n_classes;
    node_ar = new skl_tree_node[node_count];
    value_ar = new double[node_count*1*class_count](); // DAAL only supports scalar responses for now
}


bool toSKLearnRegressionTreeObjectVisitor::onSplitNode(const daal::algorithms::tree_utils::regression::SplitNodeDescriptor &desc)
{
    assert(false);
    return true;
}

bool toSKLearnRegressionTreeObjectVisitor::onLeafNode(const daal::algorithms::tree_utils::regression::LeafNodeDescriptor &desc)
{
    assert(false);
    return true;
}

TreeState _getTreeStateClassification(daal::services::interface1::SharedPtr<daal::algorithms::decision_forest::classification::interface1::Model> * model, size_t iTree, size_t n_classes)
{
    /* C++ knowledge challenge. Uncomment and try to explain.
    // First count nodes
    NodeDepthCountClassificationNodeVisitor ncv;
    (*model)->TraverseDFS(iTree, ncv);
    // then do the final tree traversal
    toSKLearnClassificationTreeObjectVisitor tsv(ncv.depth, ncv.n_nodes, ncv.n_leaf_nodes, n_classes);
    (*model)->traverseDFS(iTree, tsv);
    return TreeState(tsv);
    //*/
    return _getTreeState<ClassificationTreeNodeVisitor, ClassificationSplitNodeDescriptor, ClassificationLeafNodeDescriptor, toSKLearnClassificationTreeObjectVisitor>(model, iTree, n_classes);
}

TreeState _getTreeStateRegression(daal::services::interface1::SharedPtr<daal::algorithms::decision_forest::regression::interface1::Model> *model, size_t iTree, size_t n_classes)
{
    return _getTreeState<RegressionTreeNodeVisitor, RegressionSplitNodeDescriptor, RegressionLeafNodeDescriptor, toSKLearnRegressionTreeObjectVisitor>(model, iTree, n_classes);
}

