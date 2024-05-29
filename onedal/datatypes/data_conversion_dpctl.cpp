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
#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <utility>
#include <string>

#include "oneapi/dal/table/homogen.hpp"
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
    if ((ndim != 1) && (ndim != 2))
        report_problem_from_dptensor(err_message);
    return ndim;
}

auto get_dptensor_shape(const dpctl::tensor::usm_ndarray& tensor) {
    const auto ndim = get_and_check_dptensor_ndim(tensor);
    std::int64_t row_count, col_count;
    if (ndim == 1l) {
        row_count = dal::detail::integral_cast<std::int64_t>(tensor.get_shape(0));
        col_count = 1l;
    }
    else {
        row_count = dal::detail::integral_cast<std::int64_t>(tensor.get_shape(0));
        col_count = dal::detail::integral_cast<std::int64_t>(tensor.get_shape(1));
    }

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
    }
    else {
        //if (!is_c_cont || !is_f_cont) report_problem_from_dptensor(
        //    ": 2D array should be contiguous at least by one axis");
        return is_c_cont ? dal::data_layout::row_major : dal::data_layout::column_major;
    }
}

template <typename Type>
dal::table convert_to_homogen_impl(py::object obj, dpctl::tensor::usm_ndarray& tensor) {
    const dpctl::tensor::usm_ndarray* const ptr = &tensor;
    const auto deleter = [obj](const Type*) {
        obj.dec_ref();
    };
    const auto [r_count, c_count] = get_dptensor_shape(tensor);
    const auto layout = get_dptensor_layout(tensor);
    const auto* data = tensor.get_data<Type>();
    const auto queue = tensor.get_queue();

    auto res = dal::homogen_table(queue,
                                  data,
                                  r_count,
                                  c_count, //
                                  deleter,
                                  std::vector<sycl::event>{},
                                  layout);

    obj.inc_ref();

    return res;
}

dal::table convert_from_dptensor(py::object obj) {
    auto tensor = pybind11::cast<dpctl::tensor::usm_ndarray>(obj);

    const auto type = tensor.get_typenum();
    const auto elsize = tensor.get_elemsize();

    dal::table res{};

#define MAKE_HOMOGEN_TABLE(CType) \
    res = convert_to_homogen_impl<CType>(obj, tensor);

    SET_NPY_FEATURE(type,
                    elsize,
                    MAKE_HOMOGEN_TABLE, //
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

// TODO:
// return type.
std::string get_npy_typestr(const dal::data_type dtype) {
    switch (dtype) {
        case dal::data_type::float32: {
            return "<f4";
            break;
        }
        case dal::data_type::float64: {
            return "<f8";
            break;
        }
        case dal::data_type::int32: {
            return "<i4";
            break;
        }
        case dal::data_type::int64: {
            return "<i8";
            break;
        }
        default: report_problem_to_dptensor(": unknown data type");
    };
}

py::tuple get_npy_strides(const dal::data_layout& data_layout,
                          npy_intp row_count,
                          npy_intp column_count) {
    if (data_layout == dal::data_layout::unknown) {
        report_problem_to_dptensor(": unknown data layout");
    }
    py::tuple strides;
    if (data_layout == dal::data_layout::row_major) {
        strides = py::make_tuple(column_count, 1l);
    }
    else {
        strides = py::make_tuple(1l, row_count);
    }
    return strides;
}

py::dict construct_sua_iface(const dal::table& input) {
    const auto kind = input.get_kind();
    if (kind != dal::homogen_table::kind())
        report_problem_to_dptensor(": only homogen tables are supported");

    const auto& homogen_input = reinterpret_cast<const dal::homogen_table&>(input);
    const dal::data_type dtype = homogen_input.get_metadata().get_data_type(0);
    const dal::data_layout data_layout = homogen_input.get_data_layout();

    npy_intp row_count = dal::detail::integral_cast<npy_intp>(
        homogen_input.get_row_count());
    npy_intp column_count = dal::detail::integral_cast<npy_intp>(
        homogen_input.get_column_count());

    // need "version", "data", "shape", "typestr", "syclobj"
    py::tuple shape = py::make_tuple(row_count, column_count);
    py::list data_entry(2);

    auto bytes_array = dal::detail::get_original_data(homogen_input);
    if (!bytes_array.get_queue().has_value()) {
        report_problem_to_dptensor(": table has no queue");
    }
    auto queue = bytes_array.get_queue().value();

    const bool is_mutable = bytes_array.has_mutable_data();

    static_assert(sizeof(std::size_t) == sizeof(void*));
    data_entry[0] = is_mutable ? reinterpret_cast<std::size_t>(bytes_array.get_mutable_data())
                               : reinterpret_cast<std::size_t>(bytes_array.get_data());
    data_entry[1] = is_mutable;

    py::dict iface;
    iface["data"] = data_entry;
    iface["shape"] = shape;
    iface["strides"] = get_npy_strides(data_layout, row_count, column_count);
    // dpctl supports only version 1.
    iface["version"] = 1;
    iface["typestr"] = get_npy_typestr(dtype);
    iface["syclobj"] = py::cast(queue);

    return iface;
}

// We are using `__sycl_usm_array_interface__` attribute for constructing
// dpctl tensor on python level.
void define_sycl_usm_array_property(py::class_<dal::table>& table_obj) {
    table_obj.def_property_readonly("__sycl_usm_array_interface__", &construct_sua_iface);
}

} // namespace oneapi::dal::python

#endif // ONEDAL_DPCTL_INTEGRATION
