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

#include "prims/backend/kernel_functions_py.h"
#include "common/backend/utils.h"
#include "common/backend/compute.h"

namespace oneapi::dal::python
{
template <typename Descriptor, typename... Args>
auto linear_compute_descriptor_impl(Descriptor descriptor, const linear_kernel_params & params, Args &&... args)
{
    descriptor.set_scale(params.scale).set_shift(params.shift);
    return python::compute(descriptor, std::forward<Args>(args)...);
}

template <typename Descriptor, typename... Args>
auto rbf_compute_descriptor_impl(Descriptor descriptor, const rbf_kernel_params & params, Args &&... args)
{
    descriptor.set_sigma(params.sigma);
    return python::compute(descriptor, std::forward<Args>(args)...);
}

template <typename Descriptor, typename... Args>
auto polynomial_compute_descriptor_impl(Descriptor descriptor, const polynomial_kernel_params & params, Args &&... args)
{
    descriptor.set_scale(params.scale).set_shift(params.shift).set_degree(params.degree);
    return python::compute(descriptor, std::forward<Args>(args)...);
}

template <typename... Args>
linear_kernel::compute_result<> linear_compute_impl(linear_kernel_params & params, data_type data_type_input, Args &&... args)
{
    if (data_type_input == data_type::float32)
    {
        return linear_compute_descriptor_impl(linear_kernel::descriptor<float> {}, params, std::forward<Args>(args)...);
    }
    else
    {
        return linear_compute_descriptor_impl(linear_kernel::descriptor<double> {}, params, std::forward<Args>(args)...);
    }
}

template <typename... Args>
rbf_kernel::compute_result<> rbf_compute_impl(rbf_kernel_params & params, data_type data_type_input, Args &&... args)
{
    if (data_type_input == data_type::float32)
    {
        return rbf_compute_descriptor_impl(rbf_kernel::descriptor<float> {}, params, std::forward<Args>(args)...);
    }
    else
    {
        return rbf_compute_descriptor_impl(rbf_kernel::descriptor<double> {}, params, std::forward<Args>(args)...);
    }
}

template <typename... Args>
polynomial_kernel::compute_result<> polynomial_compute_impl(polynomial_kernel_params & params, data_type data_type_input, Args &&... args)
{
    if (data_type_input == data_type::float32)
    {
        return polynomial_compute_descriptor_impl(polynomial_kernel::descriptor<float> {}, params, std::forward<Args>(args)...);
    }
    else
    {
        return polynomial_compute_descriptor_impl(polynomial_kernel::descriptor<double> {}, params, std::forward<Args>(args)...);
    }
}

// from descriptor
linear_kernel_compute::linear_kernel_compute(linear_kernel_params * params) : params_(*params) {}

// attributes from compute_input
void linear_kernel_compute::compute(PyObject * x, PyObject * y)
{
    thread_allow _allow;
    auto x_table    = _input_to_onedal_table(x);
    auto y_table    = _input_to_onedal_table(y);
    auto data_type  = x_table.get_metadata().get_data_type(0);
    compute_result_ = linear_compute_impl(params_, data_type, x_table, y_table);
}

// attributes from compute_result
PyObject * linear_kernel_compute::get_values()
{
    return _table_to_numpy(compute_result_.get_values());
}

rbf_kernel_compute::rbf_kernel_compute(rbf_kernel_params * params) : params_(*params) {}

// attributes from compute_input
void rbf_kernel_compute::compute(PyObject * x, PyObject * y)
{
    thread_allow _allow;
    auto x_table    = _input_to_onedal_table(x);
    auto y_table    = _input_to_onedal_table(y);
    auto data_type  = x_table.get_metadata().get_data_type(0);
    compute_result_ = rbf_compute_impl(params_, data_type, x_table, y_table);
}

// attributes from compute_result
PyObject * rbf_kernel_compute::get_values()
{
    return _table_to_numpy(compute_result_.get_values());
}

polynomial_kernel_compute::polynomial_kernel_compute(polynomial_kernel_params * params) : params_(*params) {}

// attributes from compute_input
void polynomial_kernel_compute::compute(PyObject * x, PyObject * y)
{
    thread_allow _allow;
    auto x_table    = _input_to_onedal_table(x);
    auto y_table    = _input_to_onedal_table(y);
    auto data_type  = x_table.get_metadata().get_data_type(0);
    compute_result_ = polynomial_compute_impl(params_, data_type, x_table, y_table);
}

// attributes from compute_result
PyObject * polynomial_kernel_compute::get_values()
{
    return _table_to_numpy(compute_result_.get_values());
}

} // namespace oneapi::dal::python
