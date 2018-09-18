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

#ifndef _TREE_VISITOR_H_INCLUDED_
#define _TREE_VISITOR_H_INCLUDED_

#include "daal4py.h"
#include <daal.h>
#include <vector>

#define TERMINAL_NODE -1
#define NO_FEATURE -2
#define DEFAULT_DOUBLE_VALUE NaN64

// cython will convert this struct into an numpy structured array
// This is the layout that sklearn expects for its tree traversal mechanics
struct skl_tree_node {
    ssize_t left_child;
    ssize_t right_child;
    ssize_t feature;
    double threshold;
    double impurity;
    ssize_t n_node_samples;
    double weighted_n_node_samples;

    skl_tree_node()
        : left_child(TERMINAL_NODE),
          right_child(TERMINAL_NODE),
          feature(NO_FEATURE),
          threshold(DEFAULT_DOUBLE_VALUE),
          impurity(DEFAULT_DOUBLE_VALUE),
          n_node_samples(0),
          weighted_n_node_samples(0.0)
    {}
};

// our tree visitor for counting nodes
// TODO: Needs to store leaf-node response, and split-node impurity/sample_counts values
class NodeDepthCountClassificationNodeVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor
{
public:
    NodeDepthCountClassificationNodeVisitor();
    virtual bool onLeafNode(const daal::algorithms::tree_utils::classification::LeafNodeDescriptor &desc);
    virtual bool onSplitNode(const daal::algorithms::tree_utils::classification::SplitNodeDescriptor &desc);
//protected:
    size_t n_nodes;
    size_t depth;
    size_t n_leaf_nodes;
};

// equivalent for numpy arange
template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> res;
    for(T i = start; i < stop; i += step) res.push_back(i);
    return res;
}

// We only expose the minimum information to cython
struct TreeState
{
    skl_tree_node *node_ar;
    double        *value_ar;
    size_t         max_depth;
    size_t         node_count;
    size_t         leaf_count;
    size_t         class_count;
};

// our tree visitor for getting tree state
class toSKLearnTreeObjectVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor, public TreeState
{
public:
    toSKLearnTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes);
    virtual bool onSplitNode(const daal::algorithms::tree_utils::classification::SplitNodeDescriptor &desc);
    virtual bool onLeafNode(const daal::algorithms::tree_utils::classification::LeafNodeDescriptor &desc);
protected:
    size_t  node_id;
    size_t  max_n_classes;
    std::vector<ssize_t> parents;
};

// This is the function for getting the tree state which we use in cython
// we will have different model types, so it's a template
// Note: the caller will own the memory of the 2 returned arrays!
template<typename M>
TreeState _getTreeState(M & model, size_t iTree, size_t n_classes)
{
    // First count nodes
    NodeDepthCountClassificationNodeVisitor ncv;
    (*model)->traverseDFS(iTree, ncv);
    // then do the final tree traversal
    toSKLearnTreeObjectVisitor tsv(ncv.depth, ncv.n_nodes, ncv.n_leaf_nodes, n_classes);
    (*model)->traverseDFS(iTree, tsv);
    printf("DEBUG C: %zu, %zu, %zu, %zu\n", TreeState(tsv).max_depth, TreeState(tsv).node_count, TreeState(tsv).leaf_count, TreeState(tsv).class_count);
    return TreeState(tsv);
}

#endif // _TREE_VISITOR_H_INCLUDED_
