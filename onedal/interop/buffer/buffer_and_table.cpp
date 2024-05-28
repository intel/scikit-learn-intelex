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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/detail/table_iface.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"

#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/interop/buffer/common.hpp"
#include "onedal/interop/buffer/dtype_conversion.hpp"
#include "onedal/interop/buffer/buffer_and_table.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::buffer {

template <typename Type>
inline py::ssize_t to_size(Type v) {
    return detail::integral_cast<py::ssize_t>(v);
}

template <typename Type>
inline std::int64_t to_int64(Type v) {
    return detail::integral_cast<std::int64_t>(v);
}

inline auto get_shape(const dal::homogen_table& t) {
    const auto rc = to_size(t.get_row_count());
    const auto cc = to_size(t.get_column_count());
    return std::vector<py::ssize_t>{ rc, cc };
}

template <typename Type, typename Vec = std::vector<py::ssize_t>>
inline auto get_strides(const dal::homogen_table& t, const Vec& shape) {
    const auto layout = t.get_data_layout();
    const auto is_c_like = layout == data_layout::row_major;
    const auto is_f_like = layout == data_layout::column_major;

    constexpr py::ssize_t size = sizeof(Type);
    const auto raw_rc = detail::check_mul_overflow(shape.at(0), size);
    const auto raw_cc = detail::check_mul_overflow(shape.at(1), size);

    if (is_c_like) {
        return Vec{ raw_cc, size };
    }

    if (is_f_like) {
        return Vec{ size, raw_rc };
    }

    throw std::runtime_error("Unknown layout");
}

template <typename Type>
inline auto get_layout(const py::buffer_info& info, std::int64_t rc, std::int64_t cc) {
    constexpr std::int64_t one = 1l;
    constexpr std::int64_t size = sizeof(Type);

    const auto rs_raw = to_int64(info.strides.at(0));
    const auto cs_raw = to_int64(info.strides.at(1));

    const std::int64_t rs = rs_raw / size;
    const std::int64_t cs = cs_raw / size;

    const auto is_valid_rs = rs_raw == detail::check_mul_overflow(rs, size);
    const auto is_valid_cs = cs_raw == detail::check_mul_overflow(cs, size);

    if (!is_valid_rs || !is_valid_cs) {
        throw std::runtime_error("Wrong strides");
    }

    if (rs == cc && cs == one) {
        return dal::data_layout::row_major;
    }

    if (rs == one && cs == rc) {
        return dal::data_layout::column_major;
    }

    throw std::runtime_error("Unsupported layout");
}

template <typename Type>
dal::homogen_table wrap_to_homogen_table(py::buffer_info info) {
    check_buffer<Type>(info, 2ul);

    const std::int64_t rc = to_int64(info.shape.at(0));
    const std::int64_t cc = to_int64(info.shape.at(1));
    const auto dl = get_layout<Type>(info, rc, cc);

    const auto count = detail::check_mul_overflow(rc, cc);

    dal::array<Type> arr;
    if (info.readonly) {
        buf_deleter<const Type, 2ul> deleter{ std::move(info) };
        const auto* ptr = reinterpret_cast<const Type*>(info.ptr);
        auto cshared = std::shared_ptr<const Type>(ptr, std::move(deleter));
        dal::array<Type> tmp(std::move(cshared), count);
        arr = std::move(tmp);
    }
    else {
        auto* ptr = reinterpret_cast<Type*>(info.ptr);
        buf_deleter<Type, 2ul> deleter{ std::move(info) };
        auto shared = std::shared_ptr<Type>(ptr, std::move(deleter));
        dal::array<Type> tmp(std::move(shared), count);
        arr = std::move(tmp);
    }

    return dal::homogen_table::wrap(arr, rc, cc, dl);
}

template <typename Type>
inline auto get_buffer_info(const dal::homogen_table& t) {
    const auto shape = get_shape(t);
    const auto strides = get_strides<Type>(t, shape);
    auto* const raw_ptr = const_cast<void*>(t.get_data());
    const auto fmt = py::format_descriptor<Type>::format();

    constexpr py::ssize_t two = 2ul;
    constexpr std::size_t raw_size = sizeof(Type);
    constexpr auto size = static_cast<py::ssize_t>(raw_size);

    return py::buffer_info(
        /*ptr=*/raw_ptr,
        /*itemsize=*/size,
        /*format=*/fmt,
        /*ndim=*/two,
        /*shape_in=*/shape,
        /*strides_in=*/strides,
        /*readonly=*/true);
}

inline void check_policy(const dal::homogen_table& t) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto iface = detail::get_homogen_table_iface(t);
    return check_policy(iface->get_data());
#endif // ONEDAL_DATA_PARALLEL
}

template <typename Type>
py::array_t<Type> wrap_from_homogen_table_impl(const dal::homogen_table& t) {
    check_policy(t);

    using table_t = dal::homogen_table;
    auto* const tmp_tab = new table_t{ t };
    py::buffer_info info = get_buffer_info<Type>(t);
    auto capsule = py::capsule(tmp_tab, [](void* arr) -> void {
        delete reinterpret_cast<table_t*>(arr);
    });
    return py::array_t<Type>(std::move(info), std::move(capsule));
}

py::object wrap_to_homogen_table(py::buffer_info info) {
    auto dt = convert_buffer_to_dal_type(info.format);
    auto wrap_buffer = [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        auto tab = wrap_to_homogen_table<type_t>(std::move(info));
        return py::cast( std::move(tab) );
    };

    return dal::detail::dispatch_by_data_type(dt, wrap_buffer);
}

py::object wrap_to_homogen_table(py::buffer buf) {
    py::buffer_info info = buf.request();
    return wrap_to_homogen_table(std::move(info));
}

void instantiate_wrap_to_homogen_table(py::module& pm) {
    pm.def("wrap_to_homogen_table", [](py::buffer buf) {
        return wrap_to_homogen_table(std::move(buf));
    }, py::return_value_policy::take_ownership);
}

py::object wrap_from_homogen_table(const dal::homogen_table& t) {
    const auto dtype = t.get_metadata().get_data_type(0l);
    auto wrap_table = [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        return wrap_from_homogen_table_impl<type_t>(t);
    };

    return detail::dispatch_by_data_type(dtype, wrap_table);
}

void instantiate_wrap_from_homogen_table(py::module& pm) {
    constexpr const char name[] = "wrap_from_homogen_table";
    pm.def(name, [](const dal::homogen_table& t) -> py::object {
        return wrap_from_homogen_table(t);
    }, py::return_value_policy::take_ownership);
}

void instantiate_buffer_and_table(py::module& pm) {
    instantiate_wrap_from_homogen_table(pm);
    instantiate_wrap_to_homogen_table(pm);
}

} // namespace oneapi::dal::python::interop::buffer
