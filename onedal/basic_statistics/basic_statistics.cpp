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

#include "oneapi/dal/algo/basic_statistics.hpp"

#include "onedal/common.hpp"
#include "onedal/version.hpp"

#include <string>
#include <regex>
#include <map>

namespace py = pybind11;

namespace oneapi::dal::python {

#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100

namespace basic_statistics {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::basic_statistics;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, method::dense);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "by_default", ops, Float, method::by_default);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

#define RESULT_OPTION(option) { #option, dal::basic_statistics::result_options::option }

const std::map<std::string, dal::basic_statistics::result_option_id> result_option_registry {
    RESULT_OPTION(min), RESULT_OPTION(max), RESULT_OPTION(sum), RESULT_OPTION(mean),
    RESULT_OPTION(variance), RESULT_OPTION(variation), RESULT_OPTION(sum_squares),
    RESULT_OPTION(standard_deviation), RESULT_OPTION(sum_squares_centered),
    RESULT_OPTION(second_order_raw_moment)     
};

#undef RESULT_OPTION

auto get_onedal_result_options(const py::dict& params) {
    using namespace dal::basic_statistics;

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

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const py::dict& params) {
        auto desc = dal::basic_statistics::descriptor<Float,
            dal::basic_statistics::method::dense, dal::basic_statistics::task::compute>()
            .set_result_options(get_onedal_result_options(params));
        return desc;
    }
};

template <typename Policy, typename Task>
struct init_compute_ops_dispatcher {};

template <typename Policy>
struct init_compute_ops_dispatcher<Policy, dal::basic_statistics::task::compute> {
    void operator()(py::module_& m) {
        using Task = dal::basic_statistics::task::compute;
        m.def("train",
              [](const Policy& policy,
                 const py::dict& params,
                 const table& data,
                 const table& weights) {
                  using namespace dal::basic_statistics;
                  using input_t = compute_input<Task>;

                  compute_ops ops(policy, input_t{ data, weights }, params2desc{});
                  return fptype2t{ method2t{ Task{}, ops } }(params);
              });
    }
};

template <typename Policy, typename Task>
void init_partial_compute_ops(py::module& m) {
    using prev_result_t = dal::basic_statistics::partial_compute_result<Task>;
    m.def("partial_compute", [](
        const Policy& policy,
        const py::dict& params,
        const prev_result_t& prev,
        const table& data,
        const table& weights) {
            using namespace dal::basic_statistics;
            using input_t = partial_compute_input<Task>;
            partial_compute_ops ops(policy, input_t{ prev, data, weights }, params2desc{});
            return fptype2t{ method2t{ Task{}, ops } }(params);
        }
    );
}

template <typename Policy, typename Task>
void init_finalize_compute_ops(pybind11::module_& m) {
    using namespace dal::basic_statistics;
    using input_t = partial_compute_result<Task>;
    m.def("finalize_compute", [](const Policy& policy, const pybind11::dict& params, const input_t& data) {
        finalize_compute_ops ops(policy, data, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
}

template <typename Policy, typename Task>
void init_compute_ops(py::module& m) {
    init_compute_ops_dispatcher<Policy, Task>{}(m);
}

template <typename Task>
void init_compute_result(py::module_& m) {
    using namespace dal::basic_statistics;
    using result_t = compute_result<Task>;

    auto cls = py::class_<result_t>(m, "compute_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(min, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(max, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(sum, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(mean, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(variance, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(variation, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(sum_squares, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(standard_deviation, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(sum_squares_centered, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(second_order_raw_moment, result_t);
}

template <typename Task>
void init_partial_compute_result(py::module_& m) {
    using namespace dal::basic_statistics;
    using result_t = partial_compute_result<Task>;

    py::class_<result_t>(m, "partial_compute_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(partial_n_rows, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_min, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_max, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum_squares, result_t)
        .DEF_ONEDAL_PY_PROPERTY(partial_sum_squares_centered, result_t);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_compute_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_partial_compute_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_finalize_compute_ops);

} // namespace basic_statistics

ONEDAL_PY_INIT_MODULE(basic_statistics) {
    using namespace dal::detail;
    using namespace basic_statistics;
    using namespace dal::basic_statistics;

    auto sub = m.def_submodule("basic_statistics");
    using task_list = types<dal::basic_statistics::task::compute>;

#ifdef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_spmd, task_list);
#else // ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_partial_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_finalize_compute_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_compute_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_partial_compute_result, sub, task_list);
#endif // ONEDAL_DATA_PARALLEL_SPMD
}

ONEDAL_PY_TYPE2STR(dal::basic_statistics::task::compute, "compute");

#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100

} // namespace oneapi::dal::python
