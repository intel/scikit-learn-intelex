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

#include <pybind11/pybind11.h>

#include "oneapi/dal/algo/covariance.hpp"

#include "onedal/common.hpp"

namespace oneapi::dal::python {
namespace covariance {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::covariance;
        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::covariance;
        auto desc = dal::covariance::descriptor<Float, Method>{}.set_result_options(
            dal::covariance::result_options::cov_matrix | dal::covariance::result_options::means);
        return desc;
    }
};

template <typename Policy, typename Task>
void init_compute_ops(pybind11::module_& m) {
    m.def("compute", [](const Policy& policy, const pybind11::dict& params, const table& data) {
        using namespace dal::covariance;
        using input_t = compute_input<Task>;
        compute_ops ops(policy, input_t{ data }, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
}

template <typename Task>
inline void init_compute_result(pybind11::module_& m) {
    using namespace dal::covariance;
    using result_t = compute_result<Task>;
    pybind11::class_<result_t>(m, "compute_result")
        .def(pybind11::init())
        .def_property("cov_matrix", &result_t::get_cov_matrix, &result_t::set_cov_matrix)
        .def_property("means", &result_t::get_means, &result_t::set_means);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
} //namespace covariance

ONEDAL_PY_INIT_MODULE(covariance) {
    using namespace dal::detail;
    using namespace covariance;
    using namespace dal::covariance;

    auto sub = m.def_submodule("covariance");
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task::compute);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task::compute);
}

} // namespace oneapi::dal::python
