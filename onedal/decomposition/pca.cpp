#include "oneapi/dal/algo/pca.hpp"

#include "onedal/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

struct params2desc {
    template <typename Float, typename Method, typename Task>
    auto operator()(const pybind11::dict& params) {
        using namespace pca;

        //constexpr bool is_bf = std::is_same_v<Method, method::brute_force>;
        const auto n_components = params["n_components"].cast<std::int64_t>();
        //sign-flip feauture is always used in scikit-learn
        bool is_deterministic= params["is_deterministic"].cast<bool>();

        auto desc = dal::pca::descriptor<Float, Method>().set_component_count(n_components).set_deterministic(false);
        
        return desc;
    }
};

template <typename Task, typename Ops>
struct method2t {
    method2t(const Task& task, const Ops& ops) : ops(ops) {}

    template <typename Float>
    auto operator()(const py::dict& params) {
        using namespace pca;

        const auto method = params["method"].cast<std::string>();
        ONEDAL_PARAM_DISPATCH_VALUE(method, "cov", ops, Float, method::cov);
        ONEDAL_PARAM_DISPATCH_VALUE(method, "svd", ops, Float, method::svd);
        ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE(method);
    }

    Ops ops;
};

template <typename Task>
void init_model(py::module_& m) {
    using namespace pca;
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
    using namespace pca;
    using result_t = train_result<Task>;

    py::class_<result_t>(m, "train_result").def(py::init())
        .DEF_ONEDAL_PY_PROPERTY(model, result_t)
        .def_property_readonly("eigenvectors", &result_t::get_eigenvectors)
        .DEF_ONEDAL_PY_PROPERTY(eigenvalues, result_t)
        .DEF_ONEDAL_PY_PROPERTY(variances, result_t)
        .DEF_ONEDAL_PY_PROPERTY(means, result_t);
}

template <typename Task>
void init_infer_result(py::module_& m) {
    using namespace pca;
    using result_t = infer_result<Task>;

    auto cls = py::class_<result_t>(m, "infer_result")
                   .def(py::init())
                   .DEF_ONEDAL_PY_PROPERTY(transformed_data, result_t);
}

template <typename Policy, typename Task>
void init_train_ops(py::module& m) {
    m.def("train",
          [](const Policy& policy,
             const py::dict& params,
             const table& data
             //const table& responses
             ) {
              using namespace pca;
              using input_t = train_input<Task>;

              train_ops ops(policy, input_t{ data }, params2desc{} );
              return fptype2t { method2t { Task {}, ops } }(params);
          });
}

template <typename Policy, typename Task>
void init_infer_ops(py::module_& m) {
    m.def("infer",
          [](const Policy& policy,
             const py::dict& params,
             const pca::model<Task>& model,
             const table& data) {
              using namespace pca;
              using input_t = infer_input<Task>;

              infer_ops ops(policy, input_t{ model, data }, params2desc{} );
              return fptype2t { method2t { Task{ }, ops } }(params);
          });
}

ONEDAL_PY_TYPE2STR(pca::task::dim_reduction, "dim_reduction");

ONEDAL_PY_DECLARE_INSTANTIATOR(init_model);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_train_ops);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_infer_ops);

ONEDAL_PY_INIT_MODULE(decomposition) {
    using namespace pca;
    using namespace dal::detail;

    using task_list = types<task::dim_reduction>;
    auto sub = m.def_submodule("decomposition");

    ONEDAL_PY_INSTANTIATE(init_train_ops, sub, policy_list, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_ops, sub, policy_list, task_list);

    ONEDAL_PY_INSTANTIATE(init_model, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_train_result, sub, task_list);
    ONEDAL_PY_INSTANTIATE(init_infer_result, sub, task_list);
}

} //namespace oneapi::dal::python
