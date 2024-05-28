/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <pybind11/pybind11.h>

#include "onedal/common.hpp"
#include "onedal/common/dtype_dispatcher.hpp"
#include "onedal/data_management/table/table_iface.hpp"

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::data_management {

template <typename Type, typename Table = dal::homogen_table>
inline void instantiate_homogen_constructor_impl(py::class_<Table>& py_table) {
    py_table.def(py::init([](const array<Type>& arr, std::int64_t rc, std::int64_t cc) {
        return homogen_table::wrap<Type>(arr, rc, cc);
    }));
    py_table.def(
        py::init([](const array<Type>& arr, std::int64_t rc, std::int64_t cc, data_layout dl) {
            return homogen_table::wrap<Type>(arr, rc, cc, dl);
        }));
}

template <typename Table, typename... Types>
inline void instantiate_homogen_constructor(py::class_<Table>& py_table,
                                            const std::tuple<Types...>* const = nullptr) {
    static_assert(std::is_same_v<Table, homogen_table>);
    return detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            instantiate_homogen_constructor_impl<type_t>(py_table);
        },
        Types{}...);
}

inline std::int64_t get_count(const dal::homogen_table& t) {
    const std::int64_t row_count = t.get_row_count();
    const std::int64_t col_count = t.get_column_count();
    return detail::check_mul_overflow(row_count, col_count);
}

template <typename Type>
dal::array<Type> get_data(const dal::homogen_table& table) {
    const std::int64_t elem_size = //
        detail::integral_cast<std::int64_t>(sizeof(Type));

    auto impl = detail::get_homogen_table_iface(table);
    const std::int64_t count = get_count(table);

    dal::array<dal::byte_t> data = impl->get_data();

    const std::int64_t size = //
        detail::check_mul_overflow(count, elem_size);

    if (size != data.get_count()) {
        throw std::length_error("Incorrect data size");
    }

    if (data.has_mutable_data()) {
        dal::byte_t* raw_ptr = data.get_mutable_data();
        Type* ptr = reinterpret_cast<Type*>(raw_ptr);
        return dal::array<Type>(data, ptr, count);
    }
    else {
        const dal::byte_t* raw_ptr = data.get_data();
        const Type* ptr = reinterpret_cast<const Type*>(raw_ptr);
        return dal::array<Type>(data, ptr, count);
    }
}

py::object get_data_array(const dal::homogen_table& table) {
    const table_metadata& meta = table.get_metadata();
    const data_type dtype = meta.get_data_type(0l);

    return detail::dispatch_by_data_type(dtype, [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        auto array = get_data<type_t>(table);
        return py::cast(std::move(array));
    });
}

void instantiate_homogen_table(py::module& pm) {
    constexpr const char name[] = "homogen_table";
    py::class_<dal::homogen_table> py_homogen_table(pm, name);

    py_homogen_table.def(py::init<dal::table>());
    py_homogen_table.def("get_data", &get_data_array);

    instantiate_table_iface(py_homogen_table);

    constexpr const supported_types_t* types = nullptr;
    instantiate_homogen_constructor(py_homogen_table, types);
}

} // namespace oneapi::dal::python::data_management
