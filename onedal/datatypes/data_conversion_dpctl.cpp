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

#ifdef ONEDAL_DPCTL_INTEGRATION
#include <cstdio>
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

std::int64_t get_and_check_dptensor_ndim(const dpctl::tensor::usm_ndarray& tensor) {
    constexpr const char* const err_message = ": only 1D & 2D tensors are allowed";

    const auto ndim = dal::detail::integral_cast<std::int64_t>(tensor.get_ndim());
    if ((ndim != 1) && (ndim != 2)) report_problem_from_dptensor(err_message);
    return ndim;
}

auto get_dptensor_shape(const dpctl::tensor::usm_ndarray& tensor) {
    const auto ndim = get_and_check_dptensor_ndim(tensor);
    std::int64_t row_count, col_count;
    if (ndim == 1l) {
        row_count = dal::detail::integral_cast<std::int64_t>(tensor.get_shape(0));
        col_count = 1l;
    } else {
        row_count = dal::detail::integral_cast<std::int64_t>(tensor.get_shape(0));
        col_count = dal::detail::integral_cast<std::int64_t>(tensor.get_shape(1));
    }
    std::printf("Row count: %d, Col count: %d \n", int(row_count), int(col_count));
    return std::make_pair(row_count, col_count);
}

auto get_dptensor_layout(const dpctl::tensor::usm_ndarray& tensor) {
    const auto ndim = get_and_check_dptensor_ndim(tensor);
    const bool is_c_cont = tensor.is_c_contiguous();
    const bool is_f_cont = tensor.is_f_contiguous();

    if (ndim == 1l) {
        //if (!is_c_cont || !is_f_cont) report_problem_from_dptensor(
        //    ": 1D array should be contiguous both as C-order and F-order");
        return dal::data_layout::row_major;
    } else {
        //if (!is_c_cont || !is_f_cont) report_problem_from_dptensor(
        //    ": 2D array should be contiguous at least by one axis");
        return is_c_cont ? dal::data_layout::row_major : dal::data_layout::column_major;
    }
}

template<typename Type>
dal::table convert_to_homogen_impl(py::object obj, dpctl::tensor::usm_ndarray& tensor) {
    const dpctl::tensor::usm_ndarray* const ptr = &tensor;
    const auto deleter = [obj](const Type*) { obj.dec_ref(); };
    const auto [r_count, c_count] = get_dptensor_shape(tensor);
    const auto layout = get_dptensor_layout(tensor);
    const auto* data = tensor.get_data<Type>();
    const auto queue = tensor.get_queue();

    std::printf("Ready for table creation\n\n\n");

    auto res = dal::homogen_table(queue, data, r_count, c_count, //
                        deleter, std::vector<sycl::event>{}, layout);


    obj.inc_ref();

    return res;
}

dal::table convert_from_dptensor(py::object obj) {
    auto tensor = pybind11::cast<dpctl::tensor::usm_ndarray>(obj);

    const auto type = tensor.get_typenum();

    dal::table res{};

#define MAKE_HOMOGEN_TABLE(CType) \
    res = convert_to_homogen_impl<CType>(obj, tensor);

    SET_NPY_FEATURE(type, MAKE_HOMOGEN_TABLE, //
        report_problem_from_dptensor(": unknown data type"));

#undef MAKE_HOMOGEN_TABLE

    return res;
}

void report_problem_to_dptensor(const char* clarification) {
    constexpr const char* const base_message = "Unable to convert to dptensor";

    std::string message{ base_message };
    message += std::string{ clarification };
    throw std::runtime_error{ message };
}

PyObject *convert_to_dptensor(const dal::table &input) {
    return nullptr;
}

} // namespace oneapi::dal::python

#endif // ONEDAL_DPCTL_INTEGRATION
