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

#include "svm/backend/svm_py.h"
#include "common/backend/utils.h"
#include "common/backend/train.h"
#include "common/backend/infer.h"
#include "common/backend/pickling.h"

namespace oneapi::dal::python {
template <typename KernelDescriptor>
KernelDescriptor get_kernel_params(const svm_params &params) {
    if constexpr (std::is_same_v<typename KernelDescriptor::tag_t,
                                 rbf_kernel::detail::descriptor_tag>) {
        return KernelDescriptor{}.set_sigma(params.sigma);
    }

    if constexpr (std::is_same_v<typename KernelDescriptor::tag_t,
                                 polynomial_kernel::detail::descriptor_tag>) {
        return KernelDescriptor{}
            .set_scale(params.scale)
            .set_shift(params.shift)
            .set_degree(params.degree);
    }

    if constexpr (std::is_same_v<typename KernelDescriptor::tag_t,
                                 sigmoid_kernel::detail::descriptor_tag>) {
        return KernelDescriptor{}.set_scale(params.scale).set_shift(params.shift);
    }
    return KernelDescriptor{};
}

template <typename Result, typename Descriptor, typename... Args>
Result compute_descriptor_impl(Descriptor descriptor, const svm_params &params, Args &&... args) {
    using Task = typename Result::task_t;
    descriptor.set_accuracy_threshold(params.accuracy_threshold)
        .set_max_iteration_count(params.max_iteration_count)
        .set_cache_size(params.cache_size)
        .set_tau(params.tau)
        .set_shrinking(params.shrinking)
        .set_kernel(get_kernel_params<typename Descriptor::kernel_t>(params));
    if constexpr (std::is_same_v<Task, svm::task::classification>) {
        descriptor.set_class_count(params.class_count).set_c(params.c);
    }
    else if constexpr (std::is_same_v<Task, svm::task::regression>) {
        descriptor.set_epsilon(params.epsilon).set_c(params.c);
    }
    else if constexpr (std::is_same_v<Task, svm::task::nu_classification>) {
        descriptor.set_class_count(params.class_count).set_nu(params.nu);
    }
    else if constexpr (std::is_same_v<Task, svm::task::nu_regression>) {
        descriptor.set_nu(params.nu).set_c(params.c);
    }
    if constexpr (std::is_same_v<Result, typename svm::train_result<Task>>) {
        return python::train(descriptor, std::forward<Args>(args)...);
    }
    else if constexpr (std::is_same_v<Result, typename svm::infer_result<Task>>) {
        return python::infer(descriptor, std::forward<Args>(args)...);
    }
}

template <typename Result, typename... Args>
Result compute_impl(svm_params &params, data_type data_type_input, Args &&... args) {
    using Task = typename Result::task_t;
    if constexpr (std::is_same_v<Task, svm::task::classification>) {
        if (data_type_input == data_type::float32 && params.method == "smo" &&
            params.kernel == "linear") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<float, svm::method::smo, Task, linear_kernel::descriptor<float>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float32 && params.method == "smo" &&
                 params.kernel == "rbf") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<float, svm::method::smo, Task, rbf_kernel::descriptor<float>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float32 && params.method == "smo" &&
                 params.kernel == "poly") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<float,
                                svm::method::smo,
                                Task,
                                polynomial_kernel::descriptor<float>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float32 && params.method == "smo" &&
                 params.kernel == "sigmoid") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<float, svm::method::smo, Task, sigmoid_kernel::descriptor<float>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float64 && params.method == "smo" &&
                 params.kernel == "linear") {
            return compute_descriptor_impl<Result>(
                svm::
                    descriptor<double, svm::method::smo, Task, linear_kernel::descriptor<double>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float64 && params.method == "smo" &&
                 params.kernel == "rbf") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<double, svm::method::smo, Task, rbf_kernel::descriptor<double>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float64 && params.method == "smo" &&
                 params.kernel == "poly") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<double,
                                svm::method::smo,
                                Task,
                                polynomial_kernel::descriptor<double>>{},
                params,
                std::forward<Args>(args)...);
        }
        else if (data_type_input == data_type::float64 && params.method == "smo" &&
                 params.kernel == "sigmoid") {
            return compute_descriptor_impl<Result>(
                svm::descriptor<double,
                                svm::method::smo,
                                Task,
                                sigmoid_kernel::descriptor<double>>{},
                params,
                std::forward<Args>(args)...);
        }
    }

    if (data_type_input == data_type::float32 && params.method == "thunder" &&
        params.kernel == "linear") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<float, svm::method::thunder, Task, linear_kernel::descriptor<float>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float32 && params.method == "thunder" &&
             params.kernel == "rbf") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<float, svm::method::thunder, Task, rbf_kernel::descriptor<float>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float32 && params.method == "thunder" &&
             params.kernel == "poly") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<float,
                            svm::method::thunder,
                            Task,
                            polynomial_kernel::descriptor<float>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float32 && params.method == "thunder" &&
             params.kernel == "sigmoid") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<float, svm::method::thunder, Task, sigmoid_kernel::descriptor<float>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" &&
             params.kernel == "linear") {
        return compute_descriptor_impl<Result>(
            svm::
                descriptor<double, svm::method::thunder, Task, linear_kernel::descriptor<double>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" &&
             params.kernel == "rbf") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<double, svm::method::thunder, Task, rbf_kernel::descriptor<double>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" &&
             params.kernel == "poly") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<double,
                            svm::method::thunder,
                            Task,
                            polynomial_kernel::descriptor<double>>{},
            params,
            std::forward<Args>(args)...);
    }
    else if (data_type_input == data_type::float64 && params.method == "thunder" &&
             params.kernel == "sigmoid") {
        return compute_descriptor_impl<Result>(
            svm::descriptor<double,
                            svm::method::thunder,
                            Task,
                            sigmoid_kernel::descriptor<double>>{},
            params,
            std::forward<Args>(args)...);
    }
    else {
        throw std::invalid_argument("No correct parameters for onedal descriptor");
    }
}

template <typename Task>
svm_model<Task>::svm_model() {}

template <typename Task>
svm_model<Task>::svm_model(const svm::model<Task> &model) : model_(model) {}

template <typename Task>
PyObject *svm_model<Task>::serialize() {
    return serialize_si(model_);
}

template <typename Task>
void svm_model<Task>::deserialize(PyObject *py_bytes) {
    model_ = deserialize_si<svm::model<Task>>(py_bytes);
}

template <typename Task>
svm::model<Task> &svm_model<Task>::get_onedal_model() {
    return model_;
}

// from descriptor
template <typename Task>
svm_train<Task>::svm_train(svm_params *params) : params_(*params) {}

// attributes from train_input
template <typename Task>
void svm_train<Task>::train(PyObject *data, PyObject *responses, PyObject *weights) {
    auto data_table = convert_to_table(data);
    auto responses_table = convert_to_table(responses);
    auto weights_table = convert_to_table(weights);
    auto data_type = data_table.get_metadata().get_data_type(0);
    train_result_ = compute_impl<decltype(train_result_)>(params_,
                                                          data_type,
                                                          data_table,
                                                          responses_table,
                                                          weights_table);
}

// attributes from train_result
template <typename Task>
int svm_train<Task>::get_support_vector_count() {
    return train_result_.get_support_vector_count();
}

// attributes from train_result
template <typename Task>
PyObject *svm_train<Task>::get_support_vectors() {
    return convert_to_numpy(train_result_.get_support_vectors());
}

// attributes from train_result
template <typename Task>
PyObject *svm_train<Task>::get_support_indices() {
    return convert_to_numpy(train_result_.get_support_indices());
}

// attributes from train_result
template <typename Task>
PyObject *svm_train<Task>::get_coeffs() {
    return convert_to_numpy(train_result_.get_coeffs());
}

// attributes from train_result
template <typename Task>
PyObject *svm_train<Task>::get_biases() {
    return convert_to_numpy(train_result_.get_biases());
}

// attributes from train_result
template <typename Task>
svm_model<Task> svm_train<Task>::get_model() {
    return svm_model<Task>(train_result_.get_model());
}

// from descriptor
template <typename Task>
svm_infer<Task>::svm_infer(svm_params *params) : params_(*params) {}

// attributes from infer_input.hpp expect model
template <typename Task>
void svm_infer<Task>::infer(PyObject *data,
                            PyObject *support_vectors,
                            PyObject *coeffs,
                            PyObject *biases) {
    auto data_table = convert_to_table(data);
    auto support_vectors_table = convert_to_table(support_vectors);
    auto coeffs_table = convert_to_table(coeffs);
    auto biases_table = convert_to_table(biases);

    auto data_type = data_table.get_metadata().get_data_type(0);
    auto model = svm::model<Task>{}
                     .set_support_vectors(support_vectors_table)
                     .set_coeffs(coeffs_table)
                     .set_biases(biases_table);
    if constexpr (std::is_same_v<Task, svm::task::classification> ||
                  std::is_same_v<Task, svm::task::nu_classification>) {
        model.set_first_class_response(0).set_second_class_response(1);
    }
    infer_result_ = compute_impl<decltype(infer_result_)>(params_, data_type, model, data_table);
}

// attributes from infer_input.hpp expect model
template <typename Task>
void svm_infer<Task>::infer(PyObject *data, svm_model<Task> *model) {
    auto data_table = convert_to_table(data);
    auto data_type = data_table.get_metadata().get_data_type(0);
    infer_result_ = compute_impl<decltype(infer_result_)>(params_,
                                                          data_type,
                                                          model->get_onedal_model(),
                                                          data_table);
}

// attributes from infer_result
template <typename Task>
PyObject *svm_infer<Task>::get_labels() {
    return convert_to_numpy(infer_result_.get_responses());
}

// attributes from infer_result
template <typename Task>
PyObject *svm_infer<Task>::get_decision_function() {
    if constexpr (std::is_same_v<Task, svm::task::classification> ||
                  std::is_same_v<Task, svm::task::nu_classification>) {
        return convert_to_numpy(infer_result_.get_decision_function());
    }
    return nullptr;
}

template class ONEDAL_BACKEND_EXPORT svm_model<svm::task::classification>;
template class ONEDAL_BACKEND_EXPORT svm_train<svm::task::classification>;
template class ONEDAL_BACKEND_EXPORT svm_infer<svm::task::classification>;

template class ONEDAL_BACKEND_EXPORT svm_model<svm::task::regression>;
template class ONEDAL_BACKEND_EXPORT svm_train<svm::task::regression>;
template class ONEDAL_BACKEND_EXPORT svm_infer<svm::task::regression>;

template class ONEDAL_BACKEND_EXPORT svm_model<svm::task::nu_classification>;
template class ONEDAL_BACKEND_EXPORT svm_train<svm::task::nu_classification>;
template class ONEDAL_BACKEND_EXPORT svm_infer<svm::task::nu_classification>;

template class ONEDAL_BACKEND_EXPORT svm_model<svm::task::nu_regression>;
template class ONEDAL_BACKEND_EXPORT svm_train<svm::task::nu_regression>;
template class ONEDAL_BACKEND_EXPORT svm_infer<svm::task::nu_regression>;

} // namespace oneapi::dal::python
