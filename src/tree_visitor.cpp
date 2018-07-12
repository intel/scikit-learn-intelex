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

NodeDepthCountClassificationNodeVisitor::NodeDepthCountClassificationNodeVisitor()
    : n_nodes(0),
      depth(0),
      n_leaf_nodes(0)
{}

// TODO: Needs to store leaf-node response, and split-node impurity/sample_counts values
bool NodeDepthCountClassificationNodeVisitor::onLeafNode(const daal::algorithms::tree_utils::classification::LeafNodeDescriptor &desc)
{
    ++n_nodes;
    ++n_leaf_nodes;
    depth = std::max((const size_t)depth, desc.level);
    return true;
}

bool NodeDepthCountClassificationNodeVisitor::onSplitNode(const daal::algorithms::tree_utils::classification::SplitNodeDescriptor &desc)
{
    ++n_nodes;
    depth = std::max((const size_t)depth, desc.level);
    return true;
}

    
toSKLearnTreeObjectVisitor::toSKLearnTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes)
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

bool toSKLearnTreeObjectVisitor::onSplitNode(const daal::algorithms::tree_utils::classification::SplitNodeDescriptor &desc)
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

bool toSKLearnTreeObjectVisitor::onLeafNode(const daal::algorithms::tree_utils::classification::LeafNodeDescriptor &desc)
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
