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
#include "onedal/version.hpp"

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
        auto desc = dal::covariance::descriptor<Float, Method>{};
        desc.set_result_options(dal::covariance::result_options::cov_matrix | dal::covariance::result_options::means);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
        if (params.contains("bias")) {
            desc.set_bias(params["bias"].cast<bool>());
        }
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION>=20240001
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
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
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

template <typename Policy, typename Task>
void init_partial_compute_ops(pybind11::module_& m) {
    using prev_result_t = dal::covariance::partial_compute_result<Task>;
    m.def("partial_compute", [](
        const Policy& policy,
        const pybind11::dict& params,
        const prev_result_t& prev,
        const table& data) {
            using namespace dal::covariance;
            using input_t = partial_compute_input<Task>;
            partial_compute_ops ops(policy, input_t{prev, data}, params2desc{});
            return fptype2t{ method2t{ Task{}, ops } }(params);
        }
    );
}

template <typename Policy, typename Task>
void init_finalize_compute_ops(pybind11::module_& m) {
    using namespace dal::covariance;
    using input_t = partial_compute_result<Task>;
    m.def("finalize_compute", [](const Policy& policy, const pybind11::dict& params, const input_t& data) {
        finalize_compute_ops ops(policy, data, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
}


template <typename Task>
inline void init_compute_result(py::module_& m) {
    using namespace dal::covariance;
    using result_t = compute_result<Task>;
    py::class_<result_t>(m, "compute_result")
        .def(py::init())
        .def_property("cov_matrix", &result_t::get_cov_matrix, &result_t::set_cov_matrix)
        .def_property("means", &result_t::get_means, &result_t::set_means);
}

template <typename Task>
inline void init_partial_compute_result(pybind11::module_& m) {
    using namespace dal::covariance;
    using result_t = partial_compute_result<Task>;
    pybind11::class_<result_t>(m, "partial_compute_result")
        .def(pybind11::init())
        .def_property("partial_n_rows", &result_t::get_partial_n_rows, &result_t::set_partial_n_rows)
        .def_property("partial_crossproduct", &result_t::get_partial_crossproduct, &result_t::set_partial_crossproduct)
        .def_property("partial_sums", &result_t::get_partial_sum, &result_t::set_partial_sum);
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
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_finalize_compute_ops);
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
    ONEDAL_PY_INSTANTIATE(init_partial_compute_ops, sub, policy_list, task::compute); 
    ONEDAL_PY_INSTANTIATE(init_finalize_compute_ops, sub, policy_list, task::compute);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task::compute);
    ONEDAL_PY_INSTANTIATE(init_partial_compute_result, sub, task::compute);
    #if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
        ONEDAL_PY_INSTANTIATE(init_compute_hyperparameters, sub, task::compute);
    #endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240000
}

} // namespace oneapi::dal::python
