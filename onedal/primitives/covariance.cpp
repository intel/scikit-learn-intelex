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

namespace py = pybind11;

namespace oneapi::dal::python {
namespace covariance {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::covariance;
        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        using namespace dal::covariance;
        auto desc = dal::covariance::descriptor<Float, Method>{}.set_result_options(
            dal::covariance::result_options::cov_matrix | dal::covariance::result_options::means);
        return desc;
    }
};

template <typename Policy, typename Task>
void init_compute_ops(py::module_& m) {
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
    using compute_hyperparams_t = dal::covariance::detail::compute_parameters<Task>;
    m.def("compute", [](
        const Policy& policy,
        const py::dict& params,
        const compute_hyperparams_t hyperparams,
        const table& data) {
            using namespace dal::covariance;
            using input_t = compute_input<Task>;

            compute_ops_with_hyperparams ops(
                policy, input_t{ data }, params2desc{}, hyperparams);
            return fptype2t{ method2t{ Task{}, ops } }(params);
        }
    );
}
#else
    m.def("compute", [](
        const Policy& policy,
        const py::dict& params,
        const table& data) {
            using namespace dal::covariance;
            using input_t = compute_input<Task>;
            compute_ops ops(policy, input_t{ data }, params2desc{});
            return fptype2t{ method2t{ Task{}, ops } }(params);
        }
    );
}
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Task>
inline void init_compute_result(py::module_& m) {
    using namespace dal::covariance;
    using result_t = compute_result<Task>;
    py::class_<result_t>(m, "compute_result")
        .def(py::init())
        .def_property("cov_matrix", &result_t::get_cov_matrix, &result_t::set_cov_matrix)
        .def_property("means", &result_t::get_means, &result_t::set_means);
}

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

template <typename Task>
void init_compute_hyperparameters(py::module_& m) {
    using namespace dal::covariance::detail;
    using compute_hyperparams_t = compute_parameters<Task>;

    auto cls = py::class_<compute_hyperparams_t>(m, "compute_hyperparameters")
                   .def(py::init())
                   .def("set_cpu_macro_block", [](compute_hyperparams_t& self, int64_t cpu_macro_block) {
                        self.set_cpu_macro_block(cpu_macro_block);
                   })
                   .def("get_cpu_macro_block", [](const compute_hyperparams_t& self) {
                        return self.get_cpu_macro_block();
                   });
}

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
    ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_hyperparameters);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
} //namespace covariance

ONEDAL_PY_INIT_MODULE(covariance) {
    using namespace dal::detail;
    using namespace covariance;
    using namespace dal::covariance;

    auto sub = m.def_submodule("covariance");
    #ifdef ONEDAL_DATA_PARALLEL_SPMD
        ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list_spmd, task::compute);
    #else    
        ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task::compute);
    #endif
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task::compute);
    #if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
        ONEDAL_PY_INSTANTIATE(init_compute_hyperparameters, sub, task::compute);
    #endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
}

} // namespace oneapi::dal::python
