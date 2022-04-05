/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/kmeans_init.hpp"
#include "onedal/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace kmeans_init;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "random_dense", ops, Float, method::random_dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "plus_plus_dense", ops, Float, method::plus_plus_dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "parallel_plus_dense", ops, Float, method::parallel_plus_dense);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace kmeans_init;

        auto desc = descriptor<Float, Method, Task>()
                        .set_cluster_count(params["cluster_count"].cast<std::int64_t>())
                        .set_seed(params["random_state"].cast<std::int64_t>());

        return desc;
    }
};

template <typename Policy, typename Task>
void init_compute_ops(pybind11::module_& m) {
    m.def("compute",
          [](const Policy& policy, const pybind11::dict& params, const table& x) {
              using namespace kmeans_init;
              using input_t = compute_input<Task>;

              compute_ops ops (policy, input_t{x}, params2desc{});
              return fptype2t { method2t{ Task{}, ops } }(params);
          });
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace kmeans_init;
    using result_t = compute_result<Task>;

    py::class_<result_t>(m, "compute_result").def(py::init()).DEF_ONEDAL_PY_PROPERTY(centroids, result_t);
}

ONEDAL_PY_TYPE2STR(kmeans_init::task::init, "init");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);

ONEDAL_PY_INIT_MODULE(kmeans_init) {
    using namespace kmeans_init;
    using namespace dal::detail;

    using task_list = types<task::init>;
    auto sub = m.def_submodule("kmeans_init");

    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task_list);
}

} // namespace oneapi::dal::python
