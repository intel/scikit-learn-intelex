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

#include "oneapi/dal/algo/objective_function.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#include <string>
#include <regex>
#include <map>

namespace py = pybind11;

namespace oneapi::dal::python {

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100

namespace logloss_objective {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::objective_function;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense_batch", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

#define RESULT_OPTION(option) { #option, dal::objective_function::result_options::option }

const std::map<std::string, dal::objective_function::result_option_id> result_option_registry {
    RESULT_OPTION(value), RESULT_OPTION(gradient), RESULT_OPTION(hessian)
};

#undef RESULT_OPTION

auto get_onedal_result_options(const py::dict& params) {
    using namespace dal::objective_function;

    auto result_option = params["result_option"].cast<std::string>();
    result_option_id onedal_options;

    try {
        std::regex re("\\w+");
        const std::sregex_iterator last{};
        const std::sregex_iterator first( //
            result_option.begin(),
            result_option.end(),
            re);

        for (std::sregex_iterator it = first; it != last; ++it) {
            const auto str = it->str();
            const auto match = result_option_registry.find(str);
            if (match == result_option_registry.cend()) {
                ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
            } else {
                onedal_options = onedal_options | match->second;
            }
        }
    }
    catch (std::regex_error& e) {
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(result_option);
    }

    return onedal_options;
}

template <typename Float, typename Method, typename Task>
struct descriptor_creator;

template <typename Float>
struct descriptor_creator<Float,
                          objective_function::method::dense_batch,
                          objective_function::task::compute> {
    static auto get(double L1, double L2, bool intercept) {
        auto logloss_desc = logloss_objective::descriptor<Float>(L1, L2, intercept);
        return objective_function::descriptor<Float,
                                             objective_function::method::dense_batch,
                                             objective_function::task::compute>(logloss_desc);
    }
};

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        using namespace dal::linear_regression;

        const auto intercept = params["intercept"].cast<bool>();
        const double L1 = params["l1_coef"].cast<double>();
        const double L2 = params["l2_coef"].cast<double>();

        auto desc = descriptor_creator<Float, Method, Task>::get(L1, L2, intercept).set_result_options(
            get_onedal_result_options(params));
        return desc;
    }
};

template <typename Policy, typename Task>
struct init_compute_ops_dispatcher {};

template <typename Policy>
struct init_compute_ops_dispatcher<Policy, dal::objective_function::task::compute> {
    void operator()(py::module_& m) {
        using Task = dal::objective_function::task::compute;
        m.def("train", // why train?
              [](const Policy& policy,
                 const py::dict& params,
                 const table& data,
                 const table& weights,
                 const table& labels) {
                  using namespace dal::objective_function;
                  using input_t = compute_input<Task>;

                  compute_ops ops(policy, input_t{ data, weights, labels }, params2desc{});
                  return fptype2t{ method2t{ Task{}, ops } }(params);
              });
    }
};

template <typename Policy, typename Task>
void init_compute_ops(py::module& m) {
    init_compute_ops_dispatcher<Policy, Task>{}(m);
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace dal::objective_function;
    using result_t = compute_result<Task>;

    auto cls = py::class_<result_t>(m, "compute_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(value, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(gradient, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(hessian, result_t);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);

ONEDAL_PY_INIT_MODULE(logloss_objective) {
    using namespace dal::detail;
    using namespace logloss_objective;

    auto sub = m.def_submodule("logloss_objective");
    using task_list = types<dal::objective_function::task::compute>;

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD

    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task_list);
}

ONEDAL_PY_TYPE2STR(dal::objective_function::task::compute, "compute");

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100


} // namespace logloss_objective

} // namespace oneapi::dal::python