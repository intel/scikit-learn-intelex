/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL

#include "onedal/common/pybind11_helpers.hpp"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "numpy/arrayobject.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>
#include <vector>

#define TERMINAL_NODE -1
#define NO_FEATURE -2

namespace py = pybind11;

namespace dal = oneapi::dal;
namespace df = dal::decision_forest;

namespace oneapi::dal::python {

inline static const double get_nan64()
{
    return std::numeric_limits<double>::quiet_NaN();
}

#define OVERFLOW_CHECK_BY_ADDING(type, op1, op2)                                       \
    {                                                                                  \
        volatile type r = (op1) + (op2);                                               \
        r -= (op1);                                                                    \
        if (!(r == (op2))) throw std::runtime_error("Buffer size integer overflow");   \
    }

#define OVERFLOW_CHECK_BY_MULTIPLICATION(type, op1, op2)                                   \
    {                                                                                      \
        if (!(0 == (op1)) && !(0 == (op2)))                                                \
        {                                                                                  \
            volatile type r = (op1) * (op2);                                               \
            r /= (op1);                                                                    \
            if (!(r == (op2))) throw std::runtime_error("Buffer size integer overflow");   \
        }                                                                                  \
    }

// equivalent for numpy arange
template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> res;
    for(T i = start; i < stop; i += step) res.push_back(i);
    return res;
}

// This is the layout that sklearn expects for its tree traversal mechanics
class skl_tree_node {
public:
    Py_ssize_t left_child;
    Py_ssize_t right_child;
    Py_ssize_t feature;
    double threshold;
    double impurity;
    Py_ssize_t n_node_samples;
    double weighted_n_node_samples;

    skl_tree_node()
        : left_child(TERMINAL_NODE),
          right_child(TERMINAL_NODE),
          feature(NO_FEATURE),
          threshold(get_nan64()),
          impurity(get_nan64()),
          n_node_samples(0),
          weighted_n_node_samples(0.0)
    {}
};

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

// Declaration and implementation.
template <typename Task>
class NodeDepthCountNodeVisitor
{
public:
    NodeDepthCountNodeVisitor()
    : n_nodes(0),
    depth(0),
    n_leaf_nodes(0)
    {}
    bool operator()(const df::leaf_node_info<Task>& info)
    {
        ++n_nodes;
        ++n_leaf_nodes;
        depth = std::max(static_cast<const size_t>(depth), static_cast<const size_t>(info.get_level()));
        //depth = std::max((const size_t)depth, info.get_level());
        return true;
    }
    bool operator()(const df::split_node_info<Task>& info)
    {
        ++n_nodes;
        // depth = std::max((const size_t)depth, info.get_level());
        depth = std::max(static_cast<const size_t>(depth), static_cast<const size_t>(info.get_level()));
        return true;
    }

    size_t n_nodes;
    size_t depth;
    size_t n_leaf_nodes;
};

//// our tree visitor for getting tree state
template <typename Task>
class toSKLearnTreeObjectVisitor : public TreeState
{
public:
    toSKLearnTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes);
    bool operator()(const df::leaf_node_info<Task>& info);
    bool operator()(const df::split_node_info<Task>& info);
protected:
    size_t  node_id;
    size_t  max_n_classes;
    std::vector<Py_ssize_t> parents;
    void _onLeafNode(const df::leaf_node_info<Task>& info);
};

template <typename Task>
toSKLearnTreeObjectVisitor<Task>::toSKLearnTreeObjectVisitor(size_t _depth, size_t _n_nodes, size_t _n_leafs, size_t _max_n_classes)
    : node_id(0),
      parents(arange<Py_ssize_t>(-1, _depth-1))
{
    max_n_classes = _max_n_classes;
    node_count = _n_nodes;
    max_depth = _depth;
    leaf_count = _n_leafs;
    class_count = _max_n_classes;
    // TODO:
    // use pybind11 numpy primitives.
    node_ar = py::array_t<skl_tree_node>(node_count);
    //node_ar = new skl_tree_node[node_count];
    //value_ar = new double[node_count*1*class_count](); // oneDAL only supports scalar responses for now
    value_ar = py::array_t<double>(node_count*1*class_count); // oneDAL only supports scalar responses for now
}

template <typename Task>
bool toSKLearnTreeObjectVisitor<Task>::operator()(const df::split_node_info<Task>& info)
{
    if(info.get_level() > 0) {
        // has parents
        Py_ssize_t parent = parents[info.get_level() - 1];
        if(node_ar[parent].left_child > 0) {
            assert(node_ar[node_id].right_child < 0);
            node_ar[parent].right_child = node_id;
        } else {
            node_ar[parent].left_child = node_id;
        }
    }
    parents[info.get_level()] = node_id;
    node_ar[node_id].feature = info.get_feature_index();
    node_ar[node_id].threshold = info.get_feature_value();
    node_ar[node_id].impurity = info.get_impurity();
    node_ar[node_id].n_node_samples = info.get_sample_count();
    node_ar[node_id].weighted_n_node_samples = info.get_sample_count();

    // wrap-up
    ++node_id;
    return true;
}

// stuff that is done for all leaf node types
template <typename Task>
void toSKLearnTreeObjectVisitor<Task>::_onLeafNode(const df::leaf_node_info<Task>& info)
{
    if(info.get_level()) {
        Py_ssize_t parent = parents[info.get_level() - 1];
        if(node_ar[parent].left_child > 0) {
            assert(node_ar[node_id].right_child < 0);
            node_ar[parent].right_child = node_id;
        } else {
            node_ar[parent].left_child = node_id;
        }
    }

    node_ar[node_id].impurity = info.get_impurity();
    node_ar[node_id].n_node_samples = info.get_sample_count();
    node_ar[node_id].weighted_n_node_samples = info.get_sample_count();
}

template<>
bool toSKLearnTreeObjectVisitor<df::task::regression>::operator()(const df::leaf_node_info<df::task::regression>& info)
{
    _onLeafNode(info);
    OVERFLOW_CHECK_BY_MULTIPLICATION(int, node_id, class_count);
    value_ar[node_id*1*class_count] = info.get_response();

    // wrap-up
    ++node_id;
    return true;
}

template<>
bool toSKLearnTreeObjectVisitor<df::task::classification>::operator()(const df::leaf_node_info<df::task::classification>& info)
{
    if (info.get_level() > 0)
    {
        size_t depth = info.get_level() - 1;
        while (depth >= 0)
        {
            size_t id = parents[depth];
            value_ar[id*1*class_count + info.get_label()] += info.get_sample_count();
            if (depth == 0)
            {
                break;
            }
            --depth;
        }
    }
    _onLeafNode(info);
    OVERFLOW_CHECK_BY_ADDING(int, node_id*1*class_count, info.get_label());
    value_ar[node_id*1*class_count + info.get_label()] += info.get_sample_count();

    // wrap-up
    ++node_id;
    return true;
}

// This is the function for getting the tree state from a forest which we use in cython
// we will have different model types, so it's a template
// Note: the caller will own the memory of the 2 returned arrays!
template <typename Task>
TreeState _getTreeState(const df::model<Task>& model, size_t iTree, size_t n_classes)
{
    // First count nodes
    NodeDepthCountNodeVisitor<Task> ncv;
    model.traverse_depth_first(iTree, ncv);
    // then do the final tree traversal
    toSKLearnTreeObjectVisitor<Task> tsv(ncv.depth, ncv.n_nodes, ncv.n_leaf_nodes, n_classes);
    model.traverse_depth_first(iTree, tsv);

    return TreeState(tsv);
}


ONEDAL_PY_INIT_MODULE(get_tree) {
    //py::class_<skl_tree_node>(m, "skl_tree_node")
    //    .def(py::init())
    //    .def_readwrite("left_child", &skl_tree_node::left_child)
    //    .def_readwrite("right_child", &skl_tree_node::right_child)
    //    .def_readwrite("feature", &skl_tree_node::feature)
    //    .def_readwrite("threshold", &skl_tree_node::threshold)
    //    .def_readwrite("impurity", &skl_tree_node::impurity)
    //    .def_readwrite("n_node_samples", &skl_tree_node::n_node_samples)
    //    .def_readwrite("weighted_n_node_samples", &skl_tree_node::weighted_n_node_samples);

    //py::class_<TreeState>(m, "TreeState")
    //    .def(py::init())
    //    .def_readwrite("node_ar", &TreeState::node_ar)
    //    .def_readwrite("value_ar", &TreeState::value_ar)
    //    .def_readwrite("max_depth", &TreeState::max_depth)
    //    .def_readwrite("node_count", &TreeState::node_count)
    //    .def_readwrite("leaf_count", &TreeState::leaf_count)
    //    .def_readwrite("class_count", &TreeState::class_count);

    py::class_<TreeState>(m, "_get_tree_state")
    .def(py::init());
}
} // namespace oneapi::dal::python
#endif


