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
#include "oneapi/dal/algo/pca.hpp"

#include "onedal/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {
namespace decomposition {
struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace dal::pca;

        const auto n_components = params["n_components"].cast<std::int64_t>();
        // sign-flip feature is always used in scikit-learn
        bool is_deterministic = params["is_deterministic"].cast<bool>();

        auto desc = dal::pca::descriptor<Float, Method>()
                        .set_component_count(n_components)
                        .set_deterministic(is_deterministic);

        return desc;
    }
};

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace dal::pca;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "cov", ops, Float, method::cov);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "svd", ops, Float, method::svd);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "precomputed", ops, Float, method::precomputed);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Task>
void init_model(py::module_& m) {
    using namespace dal::pca;
    using model_t = model<Task>;

    auto cls = py::class_<model_t>(m, "model")
                   .def(py::init())
                   .def(py::pickle(
                       [](const model_t& m) {
                           return serialize(m);
                       },
                       [](const py::bytes& bytes) {
                           return deserialize<model_t>(bytes);
                       }))
                   .DEF_ONEDAL_PY_PROPERTY(eigenvectors, model_t);
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace dal::pca;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(model, result_t)
        .def_property_readonly("eigenvectors", &result_t::get_eigenvectors)
        .DEF_ONEDAL_PY_PROPERTY(eigenvalues, result_t)
        .DEF_ONEDAL_PY_PROPERTY(variances, result_t)
        .DEF_ONEDAL_PY_PROPERTY(means, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace dal::pca;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(transformed_data, result_t);
}

template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
    m.def("train", [](const Policy& policy, const py::dict& params, const table& data) {
        using namespace dal::pca;
        using input_t = train_input<Task>;

        train_ops ops(policy, input_t{ data }, params2desc{});
        return fptype2t{ method2t{ Task{}, ops } }(params);
    });
}

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const pca::model<Task>& model,
             const table& data) {
              using namespace dal::pca;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ model, data }, params2desc{});
              return fptype2t{ method2t{ Task{}, ops } }(params);
          });
}

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);
} // namespace decomposition
ONEDAL_PY_INIT_MODULE(decomposition) {
    using namespace decomposition;
    using namespace dal::pca;
    using namespace dal::detail;

    using task_list = types<task::dim_reduction>;
    auto sub = m.def_submodule("decomposition");

    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
}

ONEDAL_PY_TYPE2STR(oneapi::dal::pca::task::dim_reduction, "dim_reduction");
} //namespace oneapi::dal::python
