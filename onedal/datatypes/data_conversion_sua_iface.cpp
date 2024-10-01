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

#ifdef ONEDAL_DATA_PARALLEL
#define NO_IMPORT_ARRAY

#include <stdexcept>
#include <string>
#include <utility>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/common/policy_common.hpp"
#include "onedal/datatypes/data_conversion_sua_iface.hpp"
#include "onedal/datatypes/utils/dtype_conversions.hpp"
#include "onedal/datatypes/utils/dtype_dispatcher.hpp"
#include "onedal/datatypes/utils/sua_iface_helpers.hpp"

namespace oneapi::dal::python {

template <typename Type>
dal::table convert_to_homogen_impl(py::object obj) {
    auto sua_iface_dict = get_sua_interface(obj);

    const auto deleter = [obj](const Type*) {
        obj.dec_ref();
    };

    const auto ndim = get_and_check_sua_iface_ndim(sua_iface_dict);

    const auto [r_count, c_count] = get_sua_iface_shape_by_values(sua_iface_dict, ndim);

    const auto layout = get_sua_iface_layout(sua_iface_dict, r_count, c_count);

    const auto* const ptr = reinterpret_cast<const Type*>(get_sua_ptr(sua_iface_dict));
    auto syclobj = sua_iface_dict["syclobj"].cast<py::object>();
    const auto queue = get_queue_from_python(syclobj);
    bool is_readonly = is_sua_readonly(sua_iface_dict);

    dal::table res{};

    if (is_readonly) {
        res = dal::homogen_table(queue,
                                 ptr,
                                 r_count,
                                 c_count,
                                 deleter,
                                 std::vector<sycl::event>{},
                                 layout);
    }
    else {
        auto* const mut_ptr = const_cast<Type*>(ptr);
        res = dal::homogen_table(queue,
                                 mut_ptr,
                                 r_count,
                                 c_count,
                                 deleter,
                                 std::vector<sycl::event>{},
                                 layout);
    }
    obj.inc_ref();
    return res;
}

dal::table convert_from_sua_iface(py::object obj) {
    auto sua_iface_dict = get_sua_interface(obj);
    const auto type = get_sua_dtype(sua_iface_dict);

    dal::table res{};

#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType>(obj);

    SET_DAL_TYPE_FROM_DAL_TYPE(type,
                               MAKE_HOMOGEN_TABLE, //
                               report_problem_for_sua_iface(": unknown data type"));

#undef MAKE_HOMOGEN_TABLE

    return res;
}

py::dict construct_sua_iface(const dal::table& input) {
    const auto kind = input.get_kind();
    if (kind != dal::homogen_table::kind())
        report_problem_to_sua_iface(": only homogen tables are supported");

    const auto& homogen_input = reinterpret_cast<const dal::homogen_table&>(input);
    const dal::data_type dtype = homogen_input.get_metadata().get_data_type(0);
    const dal::data_layout data_layout = homogen_input.get_data_layout();

    npy_intp row_count = dal::detail::integral_cast<npy_intp>(homogen_input.get_row_count());
    npy_intp column_count = dal::detail::integral_cast<npy_intp>(homogen_input.get_column_count());

    // need "version", "data", "shape", "typestr", "syclobj"
    py::tuple shape = py::make_tuple(row_count, column_count);
    py::list data_entry(2);

    auto bytes_array = dal::detail::get_original_data(homogen_input);
    if (!bytes_array.get_queue().has_value()) {
        report_problem_to_sua_iface(": table has no queue");
    }
    auto queue = std::make_shared<sycl::queue>(bytes_array.get_queue().value());

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
    iface["typestr"] = convert_dal_to_sua_type(dtype);
    iface["syclobj"] = pack_queue(queue);

    return iface;
}

// We are using `__sycl_usm_array_interface__` attribute for constructing
// dpctl tensor on python level.
void define_sycl_usm_array_property(py::class_<dal::table>& table_obj) {
    table_obj.def_property_readonly("__sycl_usm_array_interface__", &construct_sua_iface);
}

} // namespace oneapi::dal::python

#endif // ONEDAL_DATA_PARALLEL
