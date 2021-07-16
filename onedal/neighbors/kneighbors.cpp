/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/knn.hpp"

#include "onedal/common.hpp"
#include "onedal/distances/pairwise_distances.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace knn;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "brute", ops, Float, method::brute_force);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "kd_tree", ops, Float, method::kd_tree);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);

    }

    Ops ops;
};

template <typename Ops>
struct metric2t {
    metric2t(const Ops& ops)
        : ops(ops) {}

    template <typename Float, typename Method>
    auto operator()(const py::dict& params) {
        using namespace knn;

        auto metric = params["metric"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(metric, "minkowski", ops, Float, Method, minkowski_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric, "euclidean", ops, Float, Method, minkowski_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric, "chebyshev", ops, Float, Method, chebyshev_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_VALUE(metric, "cosine", ops, Float, Method, cosine_distance::descriptor<Float>);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(metric);
    }

    Ops ops;
};

auto get_onedal_voting_mode(const py::dict& params) {
    using namespace knn;

    auto weights = params["vote_weights"].cast<std::string>();
    if (weights == "uniform") {
        return voting_mode::uniform;
    }
    else if (weights == "distance") {
        return voting_mode::distance;
    }
    else
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(weights);
}

struct params2desc {
    template <typename Float, typename Method, typename Task, typename Distance>
    auto operator()(const pybind11::dict& params) {
        using namespace knn;

        constexpr bool is_cls = std::is_same_v<Task, task::classification>;
        constexpr bool is_srch = std::is_same_v<Task, task::search>;

        constexpr bool is_bf = std::is_same_v<Method, method::brute_force>;
        constexpr bool is_kd = std::is_same_v<Method, method::kd_tree>;

        const auto class_count = params["class_count"].cast<std::int64_t>();
        const auto neighbor_count = params["neighbor_count"].cast<std::int64_t>();

        auto desc = descriptor<Float, Method, Task, Distance>(class_count, neighbor_count)
                        .set_voting_mode(get_onedal_voting_mode(params));

        if constexpr (is_bf) {
            desc.set_distance(get_distance_descriptor<Distance>(params));
        }

        return desc;
    }
};

template <typename Policy, typename Task>
void init_train_ops(py::module_& m) {
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const table& data,
             const table& responses) {
              using namespace knn;
              using input_t = train_input<Task>;

              train_ops ops(policy, input_t{ data, responses }, params2desc{} );
              return fptype2t { method2t { Task{}, metric2t{ ops } } }(params);
          });
}

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const knn::model<Task>& model,
             const table& data) {
              using namespace knn;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ data, model }, params2desc{} );
              return fptype2t { method2t { Task{}, metric2t{ ops } } }(params);
          });
}

template <typename Task>
void init_model(py::module_& m) {
    using namespace knn;
    using model_t = model<Task>;

    auto cls =
        py::class_<model_t>(m, "model")
            .def(py::init())
            .def(py::pickle(
                [](const model_t& m) {
                    return serialize(m);
                },
                [](const py::bytes& bytes) {
                    return deserialize<model_t>(bytes);
                }));
}

template <typename Task>
void init_train_result(py::module_& m) {
    using namespace knn;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result")
        .def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(model, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace knn;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(indices, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(distances, result_t)
                   .DEF_ONEDAL_PY_PROPERTY(result_options, result_t);

    constexpr bool is_cls = std::is_same_v<Task, task::classification>;
    constexpr bool is_srch = std::is_same_v<Task, task::search>;

    if constexpr (is_cls) {
        // workaround for gcc which cannot deduce setters directly passed to def_property()
        auto setter = &result_t::template set_responses<>;
        cls.def_property("responses", &result_t::get_responses, setter);
    }
}

ONEDAL_PY_TYPE2STR(knn::task::classification, "classification");
ONEDAL_PY_TYPE2STR(knn::task::search, "search");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);

ONEDAL_PY_INIT_MODULE(neighbors) {
    using namespace knn;
    using namespace dal::detail;

    using task_list =
        types<task::classification, task::search>;
    auto sub = m.def_submodule("neighbors");

    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
}

} // namespace oneapi::dal::python