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
#include "onedal/datatypes/dtype_conversions.hpp"
#include "onedal/datatypes/dtype_dispatcher.hpp"

// TODO:
// add description for the sua_iface dict.

namespace oneapi::dal::python {

/// sua utils 
dal::data_type get_sua_dtype(const py::dict& sua) {
    auto dtype = sua["typestr"].cast<std::string>();
    return convert_sua_to_dal_type(std::move(dtype));
}

py::dict get_sua_interface(const py::object& obj) {
    constexpr const char name[] = "__sycl_usm_array_interface__";
    return obj.attr(name).cast<py::dict>();
}

py::tuple get_sua_data(const py::dict& sua) {
    py::tuple result = sua["data"].cast<py::tuple>();
    if (result.size() != py::ssize_t{ 2ul }) {
        throw std::length_error("Size of \"data\" tuple should be 2");
    }
    return result;
}

std::uintptr_t get_sua_ptr(const py::dict& sua) {
    const py::tuple data = get_sua_data(sua);
    return data[0ul].cast<std::uintptr_t>();
}

bool is_sua_readonly(const py::dict& sua) {
    const py::tuple data = get_sua_data(sua);
    return data[1ul].cast<bool>();
}

py::tuple get_sua_shape(const py::dict& sua) {
    py::tuple shape = sua["shape"].cast<py::tuple>();
    if (shape.size() == py::ssize_t{ 0ul }) {
        throw std::runtime_error("Wrong number of dimensions");
    }
    return shape;
}

void report_problem_for_sua_iface(const char* clarification) {
    constexpr const char* const base_message = "Unable to convert from SUA interface.";

    std::string message{ base_message };
    message += std::string{ clarification };
    throw std::invalid_argument{ message };
}

std::int64_t get_and_check_sua_iface_ndim(const py::dict& sua_dict) {
    constexpr const char* const err_message = ": only 1D & 2D tensors are allowed";
    py::tuple shape = get_sua_shape(sua_dict);
    const py::ssize_t raw_ndim = shape.size();
    const auto ndim = detail::integral_cast<std::int64_t>(raw_ndim);
    if ((ndim != 1l) && (ndim != 2l))
        report_problem_for_sua_iface(err_message);
    return ndim;
}

auto get_sua_iface_shape_by_values(const py::dict sua_dict, const std::int64_t ndim) {
    std::int64_t row_count, col_count;
    auto shape = sua_dict["shape"].cast<py::tuple>();
    if (ndim == 1l) {
        row_count = shape[0l].cast<std::int64_t>();
        col_count = 1l;
    }
    else {
        row_count = shape[0l].cast<std::int64_t>();
        col_count = shape[1l].cast<std::int64_t>();
    }
    return std::make_pair(row_count, col_count);
}

dal::data_layout get_sua_iface_layout(const py::dict& sua_dict,
                                      const std::int64_t& r_count,
                                      const std::int64_t& c_count) {
    const auto raw_strides = sua_dict["strides"];
    if (raw_strides.is_none()) {
        // None to indicate a C-style contiguous array.
        return dal::data_layout::row_major;
    }
    auto strides_tuple = raw_strides.cast<py::tuple>();

    auto strides_len = py::len(strides_tuple);

    if (strides_len == 1l) {
        return dal::data_layout::row_major;
    }
    else if (strides_len == 2l) {
        auto r_strides = strides_tuple[0l].cast<std::int64_t>();
        auto c_strides = strides_tuple[1l].cast<std::int64_t>();
        using shape_t = std::decay_t<decltype(r_count)>;
        using stride_t = std::decay_t<decltype(r_strides)>;
        constexpr auto one = static_cast<shape_t>(1);
        static_assert(std::is_same_v<shape_t, stride_t>);
        if (r_strides == c_count && c_strides == one) {
            return dal::data_layout::row_major;
        }
        else if (r_strides == one && c_strides == r_count) {
            return dal::data_layout::column_major;
        }
        else {
            throw std::runtime_error("Wrong strides");
        }
    }
    else {
        throw std::runtime_error("Unsupporterd data shape.`");
    }
}

template <typename Type>
dal::table convert_to_homogen_impl(py::object obj) {
    auto sua_iface_dict = get_sua_interface(obj);

    const auto deleter = [obj](const Type*) {
        obj.dec_ref();
    };

    const auto ndim = get_and_check_sua_iface_ndim(sua_iface_dict);

    const auto [r_count, c_count] =
        get_sua_iface_shape_by_values(sua_iface_dict, ndim);

    const auto layout = get_sua_iface_layout(sua_iface_dict, r_count, c_count);

    const auto* const ptr = reinterpret_cast<const Type*>(get_sua_ptr(sua_iface_dict));
    auto syclobj = sua_iface_dict["syclobj"].cast<py::object>();
    const auto queue = get_queue_from_python(syclobj);

    // TODO:
    // if (is_readonly) {
    //     //
    // }
    // else {
    //     // auto* const mut_ptr = const_cast<Type*>(ptr);
    // }

    auto res = dal::homogen_table(queue,
                                  ptr,
                                  r_count,
                                  c_count, //
                                  deleter,
                                  std::vector<sycl::event>{},
                                  layout);

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

void report_problem_to_sua_iface(const char* clarification) {
    constexpr const char* const base_message = "Unable to convert to SUA interface.";

    std::string message{ base_message };
    message += std::string{ clarification };
    throw std::runtime_error{ message };
}

// TODO:
// re-use convert_dal_to_sua_type instead.
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
        default: report_problem_to_sua_iface(": unknown data type");
    };
}

py::tuple get_npy_strides(const dal::data_layout& data_layout,
                          npy_intp row_count,
                          npy_intp column_count) {
    if (data_layout == dal::data_layout::unknown) {
        report_problem_to_sua_iface(": unknown data layout");
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

py::capsule pack_queue(const std::shared_ptr<sycl::queue>& queue) {
    static const char queue_capsule_name[] = "SyclQueueRef";
    if (queue.get() == nullptr) {
        throw std::runtime_error("Empty queue");
    }
    else {
        void (*deleter)(void*) = [](void* const queue) -> void {
            delete reinterpret_cast<sycl::queue* const>(queue);
        };

        sycl::queue* ptr = new sycl::queue{ *queue };
        void* const raw = reinterpret_cast<void*>(ptr);

        py::capsule capsule(raw, deleter);
        capsule.set_name(queue_capsule_name);
        return capsule;
    }
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
    iface["typestr"] = get_npy_typestr(dtype);
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
