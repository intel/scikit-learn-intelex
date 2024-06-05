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

#include "oneapi/dal/algo/louvain.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#include <regex>

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace preview::louvain;

        const auto method = params["method"].cast<std::string>();

        ONEDAL_PARAM_DISPATCH_VALUE(method, "fast", ops, Float, method::fast);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::preview::louvain;

        auto desc = descriptor<Float, Method, Task>();
        desc.set_accuracy_threshold(params["accuracy_threshold"].cast<double>());
        desc.set_resolution(params["resolution"].cast<double>());
        desc.set_max_iteration_count(params["max_iteration_count"].cast<std::int64_t>());

        return desc;
    }
};

template <typename Float>
using graph_t = dal::preview::undirected_adjacency_vector_graph<std::int32_t, Float>;


template <typename Task>
void init_vertex_partitioning_ops(py::module_& m) {
    m.def("vertex_partitioning",
          [](const py::dict& params,
             const graph_t<float>& data,
             const table& initial_partition) {
              using namespace preview::louvain;
              using input_t = vertex_partitioning_input<graph_t<float>, Task>;

              vertex_partitioning_ops ops(input_t{ data, initial_partition}, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
    m.def("vertex_partitioning",
          [](const py::dict& params,
             const graph_t<float>& data) {
              using namespace preview::louvain;
              using input_t = vertex_partitioning_input<graph_t<float>, Task>;

              vertex_partitioning_ops ops(input_t{ data}, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_vertex_partitioning_result(py::module_& m) {
    using namespace preview::louvain;
    using result_t = vertex_partitioning_result<Task>;

    py::class_<result_t>(m, "vertex_paritioning_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(labels, result_t)
        .DEF_ONEDAL_PY_PROPERTY(modularity, result_t)
        .DEF_ONEDAL_PY_PROPERTY(community_count, result_t);
}

ONEDAL_PY_TYPE2STR(preview::louvain::task::vertex_partitioning, "vertex_partitioning");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_vertex_partitioning_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_vertex_partitioning_result);

ONEDAL_PY_INIT_MODULE(louvain) {
    using namespace dal::detail;
    using namespace dal::preview::louvain;

    using task_list = types<task::vertex_partitioning>;
    using graph_list = types<dal::preview::undirected_adjacency_vector_graph<>>;
    auto sub = m.def_submodule("louvain");

    ONEDAL_PY_INSTANTIATE(init_vertex_partitioning_ops, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_vertex_partitioning_result, sub, task_list);

}

} // namespace oneapi::dal::python
