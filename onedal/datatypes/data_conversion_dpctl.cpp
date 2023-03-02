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

#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <utility>
#include <string>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/csr.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/datatypes/data_conversion_dpctl.hpp"
#include "onedal/datatypes/numpy_helpers.hpp"

#include "dpctl4pybind11.hpp"

namespace oneapi::dal::python {

void report_problem_from_dptensor(const char* clarification) {
    constexpr const char* const base_message = "Unable to convert from dptensor"; 

    std::string message{ base_message };
    message += std::string{ clarification };
    throw std::invalid_argument{ message };
}

std::int64_t get_and_check_dptensor_ndim(dpctl::tensor::usm_ndarray* ptr) {
    constexpr const char* const err_message = ": only 1D & 2D tensors are allowed";
    const auto ndim = dal::detail::integral_cast<std::int64_t>(ptr->get_ndim());
    if ((ndim != 1) && (ndim != 2)) report_problem_from_dptensor(err_message);
    return ndim;
}

auto get_dptensor_shape(dpctl::tensor::usm_ndarray* ptr) {
    const auto ndim = get_and_check_dptensor_ndim(ptr);
    const std::int64_t row_count = dal::detail::integral_cast<std::int64_t>(ptr->get_shape(0));
    const std::int64_t col_count = (ndim == 1l) ? 1l : dal::detail::integral_cast<std::int64_t>(ptr->get_shape(1));
    return std::make_pair(row_count, col_count);
}

auto get_dptensor_layout(dpctl::tensor::usm_ndarray* ptr) {
    const bool is_c_ordered = ptr->is_c_contiguous();
    const bool is_f_ordered = ptr->is_f_contiguous();
}

dal::table convert_from_dptensor(PyObject *obj) {
    using tensor = dpctl::tensor::usm_ndarray;
    auto* const ptr = dynamic_cast<tensor*>(obj);
    if(ptr == nullptr) {
        report_problem_from_dptensor(": pointer casting");
    }
    else {
        const auto typenum = ptr->get_typenum();
        const auto ndim = ptr->get_ndim();
        auto shape = ptr->get_shape_vector();

        const bool is_c_ordered = ptr->is_c_contiguous();
        const bool is_f_ordered = ptr->is_f_contiguous();


        return dal::table{};
    }
}

void report_problem_to_dptensor(const char* clarification) {
    std::string message{ "Unable to convert to dptensor: " };
    message += std::string{ clarification };
    throw std::runtime_error{ message };
}

PyObject *convert_to_dptensor(const dal::table &input) {
    return nullptr;
}

} // namespace oneapi::dal::python
