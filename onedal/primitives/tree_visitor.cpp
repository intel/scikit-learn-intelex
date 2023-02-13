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

#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/common.hpp"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "numpy/arrayobject.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <limits>
#include <vector>

#include <iostream>
#include <utility>

#define TERMINAL_NODE -1
#define NO_FEATURE    -2

namespace py = pybind11;

namespace dal = oneapi::dal;
namespace df = dal::decision_forest;

namespace oneapi::dal::python {

inline static const double get_nan64() {
    return std::numeric_limits<double>::quiet_NaN();
}

#define OVERFLOW_CHECK_BY_ADDING(type, op1, op2)                      \
    {                                                                 \
        volatile type r = (op1) + (op2);                              \
        r -= (op1);                                                   \
        if (!(r == (op2)))                                            \
            throw std::runtime_error("Buffer size integer overflow"); \
    }

#define OVERFLOW_CHECK_BY_MULTIPLICATION(type, op1, op2)                  \
    {                                                                     \
        if (!(0 == (op1)) && !(0 == (op2))) {                             \
            volatile type r = (op1) * (op2);                              \
            r /= (op1);                                                   \
            if (!(r == (op2)))                                            \
                throw std::runtime_error("Buffer size integer overflow"); \
        }                                                                 \
    }

// equivalent for numpy arange
template <typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> res;
    for (T i = start; i < stop; i += step)
        res.push_back(i);
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
              weighted_n_node_samples(0.0) {}
};

// We only expose the minimum information to python
template <typename T>
struct tree_state {
    py::array_t<skl_tree_node> node_ar;
    py::array_t<double> value_ar;
    size_t max_depth;
    size_t node_count;
    size_t leaf_count;
    size_t class_count;
};

// Declaration and implementation.
template <typename Task>
class node_count_visitor {
public:
    node_count_visitor() : n_nodes(0), depth(0), n_leaf_nodes(0) {}
    bool call(const df::leaf_node_info<Task>& info) {
        ++n_nodes;
        ++n_leaf_nodes;
        depth = std::max(depth, static_cast<const size_t>(info.get_level()));
        return true;
    }
    bool call(const df::split_node_info<Task>& info) {
        ++n_nodes;
        depth = std::max(depth, static_cast<const size_t>(info.get_level()));
        return true;
    }

    size_t n_nodes;
    size_t depth;
    size_t n_leaf_nodes;
};

template <typename Task, typename Impl>
class node_visitor {
public:
    Impl* const p_impl;
    node_visitor(Impl* impl) : p_impl{ impl } {
        assert(impl != nullptr);
    }

    node_visitor(node_visitor&&) = default;

    bool operator()(const df::leaf_node_info<Task>& info) {
        return p_impl->call(info);
    }
    bool operator()(const df::split_node_info<Task>& info) {
        return p_impl->call(info);
    }
};

// our tree visitor for getting tree state
template <typename Task>
class to_sklearn_tree_object_visitor : public tree_state<Task> {
public:
    to_sklearn_tree_object_visitor(size_t _depth,
                                   size_t _n_nodes,
                                   size_t _n_leafs,
                                   size_t _max_n_classes);
    bool call(const df::leaf_node_info<Task>& info);
    bool call(const df::split_node_info<Task>& info);

protected:
    size_t node_id;
    size_t max_n_classes;
    std::vector<Py_ssize_t> parents;
    void _onLeafNode(const df::leaf_node_info<Task>& info);
};

template <typename Task>
to_sklearn_tree_object_visitor<Task>::to_sklearn_tree_object_visitor(size_t _depth,
                                                                     size_t _n_nodes,
                                                                     size_t _n_leafs,
                                                                     size_t _max_n_classes)
        : node_id(0),
          parents(arange<Py_ssize_t>(-1, _depth - 1)) {
    this->max_n_classes = _max_n_classes;
    this->node_count = _n_nodes;
    this->max_depth = _depth;
    this->leaf_count = _n_leafs;
    this->class_count = _max_n_classes;

    auto node_ar_shape = py::array::ShapeContainer({ this->node_count });
    auto node_ar_strides = py::array::StridesContainer({ sizeof(skl_tree_node) });

    auto value_ar_shape = py::array::ShapeContainer({ static_cast<Py_ssize_t>(this->node_count),
                                                      1,
                                                      static_cast<Py_ssize_t>(this->class_count) });
    auto value_ar_strides = py::array::StridesContainer(
        { this->class_count * sizeof(double), this->class_count * sizeof(double), sizeof(double) });

    skl_tree_node* node_ar_ptr = new skl_tree_node[this->node_count];
    double* value_ar_ptr =
        new double[this->node_count * 1 *
                   this->class_count](); // oneDAL only supports scalar responses for now

    this->node_ar = py::array_t<skl_tree_node>(node_ar_shape, node_ar_strides, node_ar_ptr);
    this->value_ar = py::array_t<double>(value_ar_shape, value_ar_strides, value_ar_ptr);
}

template <typename Task>
bool to_sklearn_tree_object_visitor<Task>::call(const df::split_node_info<Task>& info) {
    py::buffer_info node_ar_buf = this->node_ar.request();

    skl_tree_node* node_ar_ptr = static_cast<skl_tree_node*>(node_ar_buf.ptr);

    if (info.get_level() > 0) {
        // has parents
        Py_ssize_t parent = parents[info.get_level() - 1];
        if (node_ar_ptr[parent].left_child > 0) {
            assert(node_ar_ptr[node_id].right_child < 0);
            node_ar_ptr[parent].right_child = node_id;
        }
        else {
            node_ar_ptr[parent].left_child = node_id;
        }
    }
    parents[info.get_level()] = node_id;
    node_ar_ptr[node_id].feature = info.get_feature_index();
    node_ar_ptr[node_id].threshold = info.get_feature_value();
    node_ar_ptr[node_id].impurity = info.get_impurity();
    node_ar_ptr[node_id].n_node_samples = info.get_sample_count();
    node_ar_ptr[node_id].weighted_n_node_samples = info.get_sample_count();

    // wrap-up
    ++node_id;
    return true;
}

// stuff that is done for all leaf node types
template <typename Task>
void to_sklearn_tree_object_visitor<Task>::_onLeafNode(const df::leaf_node_info<Task>& info) {
    py::buffer_info node_ar_buf = this->node_ar.request();

    skl_tree_node* node_ar_ptr = static_cast<skl_tree_node*>(node_ar_buf.ptr);

    if (info.get_level()) {
        Py_ssize_t parent = parents[info.get_level() - 1];
        if (node_ar_ptr[parent].left_child > 0) {
            assert(node_ar_ptr[node_id].right_child < 0);
            node_ar_ptr[parent].right_child = node_id;
        }
        else {
            node_ar_ptr[parent].left_child = node_id;
        }
    }

    node_ar_ptr[node_id].impurity = info.get_impurity();
    node_ar_ptr[node_id].n_node_samples = info.get_sample_count();
    node_ar_ptr[node_id].weighted_n_node_samples = info.get_sample_count();
}

template <>
bool to_sklearn_tree_object_visitor<df::task::regression>::call(
    const df::leaf_node_info<df::task::regression>& info) {
    _onLeafNode(info);
    OVERFLOW_CHECK_BY_MULTIPLICATION(int, node_id, class_count);

    py::buffer_info value_ar_buf = this->value_ar.request();
    double* value_ar_ptr = static_cast<double*>(value_ar_buf.ptr);

    value_ar_ptr[node_id * 1 * this->class_count] = info.get_response();

    // wrap-up
    ++node_id;
    return true;
}

template <>
bool to_sklearn_tree_object_visitor<df::task::classification>::call(
    const df::leaf_node_info<df::task::classification>& info) {
    py::buffer_info value_ar_buf = this->value_ar.request();
    double* value_ar_ptr = static_cast<double*>(value_ar_buf.ptr);

    if (info.get_level() > 0) {
        size_t depth = static_cast<const size_t>(info.get_level()) - 1;
        while (depth >= 0) {
            size_t id = parents[depth];
            value_ar_ptr[id * 1 * this->class_count + info.get_response()] +=
                info.get_sample_count();
            if (depth == 0) {
                break;
            }
            --depth;
        }
    }
    _onLeafNode(info);
    OVERFLOW_CHECK_BY_ADDING(int, node_id * 1 * this->class_count, info.get_response());
    value_ar_ptr[node_id * 1 * this->class_count + info.get_response()] += info.get_sample_count();

    // wrap-up
    ++node_id;
    return true;
}

template <typename Task>
void init_get_tree_state(py::module_& m) {
    using namespace decision_forest;
    using model_t = model<Task>;
    using tree_state_t = tree_state<Task>;

    // TODO:
    // create one instance for cls and reg.
    py::class_<tree_state_t>(m, "get_tree_state")
        .def(py::init([](const model_t& model, size_t iTree, size_t n_classes) {
            // First count nodes
            node_count_visitor<Task> ncv;
            node_visitor<Task, decltype(ncv)> ncv_decorator{ &ncv };

            model.traverse_depth_first(iTree, std::move(ncv_decorator));
            // then do the final tree traversal
            to_sklearn_tree_object_visitor<Task> tsv(ncv.depth,
                                                     ncv.n_nodes,
                                                     ncv.n_leaf_nodes,
                                                     n_classes);
            node_visitor<Task, decltype(tsv)> tsv_decorator{ &tsv };
            model.traverse_depth_first(iTree, std::move(tsv_decorator));
            return tree_state_t(tsv);
        }))
        .def_readwrite("node_ar", &tree_state_t::node_ar, py::return_value_policy::take_ownership)
        .def_readwrite("value_ar", &tree_state_t::value_ar, py::return_value_policy::take_ownership)
        .def_readwrite("max_depth", &tree_state_t::max_depth)
        .def_readwrite("node_count", &tree_state_t::node_count)
        .def_readwrite("leaf_count", &tree_state_t::leaf_count)
        .def_readwrite("class_count", &tree_state_t::class_count);
}

ONEDAL_PY_TYPE2STR(decision_forest::task::classification, "classification");
ONEDAL_PY_TYPE2STR(decision_forest::task::regression, "regression");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_get_tree_state);

ONEDAL_PY_INIT_MODULE(get_tree) {
    using namespace decision_forest;
    using namespace dal::detail;

    PYBIND11_NUMPY_DTYPE(skl_tree_node,
                         left_child,
                         right_child,
                         feature,
                         threshold,
                         impurity,
                         n_node_samples,
                         weighted_n_node_samples);

    using task_list = types<task::classification, task::regression>;
    auto sub = m.def_submodule("get_tree");

    ONEDAL_PY_INSTANTIATE(init_get_tree_state, sub, task_list);
}
} // namespace oneapi::dal::python
