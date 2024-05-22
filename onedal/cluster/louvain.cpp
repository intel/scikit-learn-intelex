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
        using namespace dbscan;

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
        using namespace dal::dbscan;

        const auto min_observations = params["accuracy_threshold"].cast<std::double>();
        const auto resolution = params["resolution"].cast<double>();
        const auto observation_count = params["observation_count"].cast<std::int64_t>();
        auto desc = descriptor<Float, Method, Task>(epsilon, min_observations);
        desc.set_mem_save_mode(params["mem_save_mode"].cast<std::int64_t>());
        desc.set_result_options(get_onedal_result_options(params));

        return desc;
    }
};

template <typename Policy, typename Task>
void init_compute_ops(py::module_& m) {
    m.def("vertex_partioning",
          [](const Policy& policy,
             const py::dict& params,
             const table& data,
             const table& weights) {
              using namespace dbscan;
              using input_t = compute_input<Task>;

              compute_ops ops(policy, input_t{ data, weights }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace dbscan;
    using result_t = compute_result<Task>;

    py::class_<result_t>(m, "compute_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(core_observations, result_t)
        .DEF_ONEDAL_PY_PROPERTY(responses, result_t)
        .DEF_ONEDAL_PY_PROPERTY(core_flags, result_t)
        .DEF_ONEDAL_PY_PROPERTY(core_observation_indices, result_t)
        .DEF_ONEDAL_PY_PROPERTY(result_options, result_t)
        .DEF_ONEDAL_PY_PROPERTY(cluster_count, result_t);
}

ONEDAL_PY_TYPE2STR(dbscan::task::clustering, "clustering");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);

// TODO:
// change the name of modue for all algos -> cluster.
ONEDAL_PY_INIT_MODULE(dbscan) {
    using namespace dal::detail;
    using namespace dbscan;
    using namespace dal::dbscan;

    using task_list = types<task::clustering>;
    auto sub = m.def_submodule("dbscan");

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD

}

} // namespace oneapi::dal::python
