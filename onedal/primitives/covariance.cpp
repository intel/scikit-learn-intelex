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

template <typename Dense, typename Ops>
struct cov_method2t {
    cov_method2t(const Dense& dense_method, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const pybind11::dict& params) {
        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "dense", ops, Float, Dense);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <template <typename F, typename M, typename T> typename Desc>
struct cov_params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        auto desc = dal::covariance::descriptor{}.set_result_options(
            dal::covariance::result_options::cov_matrix | dal::covariance::result_options::means);
        return desc;
    }
};

template <typename Policy,
          typename Input,
          typename Result,
          typename Param2Desc,
          typename DenseMethod>
inline void init_cov_compute_ops(pybind11::module_& m) {
    m.def("compute", [](const Policy& policy, const pybind11::dict& params, const table& x) {
        compute_ops ops(policy, Input{ x }, Param2Desc{});
        return fptype2t{ cov_method2t{ DenseMethod{}, ops } }(params);
    });
}

template <typename Result>
inline void init_cov_result(pybind11::module_& m) {
    pybind11::class_<Result>(m, "cov_result")
        .def(pybind11::init())
        .def_property("cov_matrix", &Result::get_cov_matrix, &Result::set_cov_matrix);
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_cov_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_cov_compute_ops);

ONEDAL_PY_INIT_MODULE(covariance) {
    using namespace dal::detail;
    using namespace covariance;
    using input_t = compute_input<task::compute>;
    using result_t = compute_result<task::compute>;
    using param2desc_t = cov_params2desc<descriptor>;

    auto sub = m.def_submodule("covariance");
    ONEDAL_PY_INSTANTIATE(init_cov_result, sub, result_t);
    ONEDAL_PY_INSTANTIATE(init_cov_compute_ops,
                          sub,
                          policy_list,
                          input_t,
                          result_t,
                          param2desc_t,
                          method::dense);
}

} // namespace oneapi::dal::python
