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

#include <memory>

#include <pybind11/pybind11.h>

#include "onedal/common.hpp"
#include "onedal/common/dtype_dispatcher.hpp"
#include "onedal/data_management/table/table_iface.hpp"

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::data_management {

template <typename Type>
inline void make_array_setter(py::class_<dal::heterogen_table>& table) {
    using array_t = dal::array<Type>;
    table.def("set_column", [](dal::heterogen_table& table, std::int64_t col, array_t arr) {
        auto as_chunked = dal::chunked_array<Type>{ std::move(arr) };
        table.set_column(col, std::move(as_chunked));
    });
}

template <typename Type>
inline void make_chunked_array_setter(py::class_<dal::heterogen_table>& table) {
    using chunked_array_t = dal::chunked_array<Type>;
    table.def("set_column", [](dal::heterogen_table& table, std::int64_t col, chunked_array_t arr) {
        table.set_column(col, std::move(arr));
    });
}

template <typename Table, typename... Types>
void instantiate_setters(py::class_<Table>& table, const std::tuple<Types...>* const = nullptr) {
    return detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            make_chunked_array_setter<type_t>(table);
            make_array_setter<type_t>(table);
        },
        Types{}...);
}

void instantiate_heterogen_table(py::module& pm) {
    constexpr const char name[] = "heterogen_table";
    py::class_<dal::heterogen_table> py_heterogen_table(pm, name);

    py_heterogen_table.def(py::init<dal::table>());
    py_heterogen_table.def(py::init([](const dal::table_metadata& meta) -> dal::heterogen_table {
        return dal::heterogen_table::empty(meta);
    }));

    instantiate_table_iface(py_heterogen_table);

    constexpr const supported_types_t* types = nullptr;
    instantiate_setters(py_heterogen_table, types);
}

} // namespace oneapi::dal::python::data_management
