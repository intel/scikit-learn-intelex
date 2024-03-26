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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::data_management {

inline void instantiate_csr_indexing(py::module& pm) {
    constexpr const char name[] = "sparse_indexing";
    py::enum_<dal::sparse_indexing> py_indexing(pm, name);
    py_indexing.value("one_based", dal::sparse_indexing::one_based);
    py_indexing.value("zero_based", dal::sparse_indexing::zero_based);
    py_indexing.export_values();
}

template <typename Type, typename Table = dal::csr_table>
inline void instantiate_csr_constructor_impl(py::class_<Table>& py_table) {
    py_table.def(py::init([](const dal::array<Type>& non_zeros,
                             const dal::array<std::int64_t>& col_indices,
                             const dal::array<std::int64_t>& row_offsets,
                             std::int64_t col_count) {
        return csr_table::wrap<Type>(non_zeros, col_indices, row_offsets, col_count);
    }));
    py_table.def(py::init([](const dal::array<Type>& non_zeros,
                             const dal::array<std::int64_t>& col_indices,
                             const dal::array<std::int64_t>& row_offsets,
                             std::int64_t col_count,
                             dal::sparse_indexing sp) {
        return csr_table::wrap<Type>(non_zeros, col_indices, row_offsets, col_count, sp);
    }));
}

template <typename Table, typename... Types>
inline void instantiate_csr_constructor(py::class_<Table>& py_table,
                                        const std::tuple<Types...>* const = nullptr) {
    static_assert(std::is_same_v<Table, csr_table>);
    return detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            instantiate_csr_constructor_impl<type_t>(py_table);
        },
        Types{}...);
}

detail::csr_table_iface* get_csr_table_iface_impl(detail::table_iface* table) {
    return dynamic_cast<detail::csr_table_iface*>(table);
}

template <typename Table>
inline std::shared_ptr<detail::csr_table_iface> get_interface(const Table& table) {
    auto pimpl = detail::pimpl_accessor{}.get_pimpl(table);
    auto csr_iface_ptr = get_csr_table_iface_impl(pimpl.get());
    return std::shared_ptr<detail::csr_table_iface>{ pimpl, csr_iface_ptr };
}

template <typename Type>
dal::array<Type> get_data(const dal::csr_table& table) {
    const std::int64_t elem_size = //
        detail::integral_cast<std::int64_t>(sizeof(Type));

    auto impl = get_interface(table);
    const auto count = table.get_non_zero_count();

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

py::object get_data_array(const dal::csr_table& table) {
    const table_metadata& meta = table.get_metadata();
    const data_type dtype = meta.get_data_type(0l);

    return detail::dispatch_by_data_type(dtype, [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        auto array = get_data<type_t>(table);
        return py::cast(std::move(array));
    });
}

void instantiate_csr_table(py::module& pm) {
    instantiate_csr_indexing(pm);

    constexpr const char name[] = "csr_table";
    py::class_<dal::csr_table> py_csr_table(pm, name);

    //py_csr_table.def(py::init<dal::table>());

    // Need to be fixed. No such symbol in the library
    py_csr_table.def(py::init([](const dal::table& t) {
        if (t.get_kind() != csr_table::kind()) {
            throw std::runtime_error("Unable to cast to csr table");
        }
        const auto& casted = reinterpret_cast<const dal::csr_table&>(t);
        return dal::csr_table(casted);
    }));

    py_csr_table.def("get_indexing", &dal::csr_table::get_indexing);
    py_csr_table.def("get_non_zero_count", [](const dal::csr_table& table) -> std::int64_t {
        return table.get_non_zero_count();
    });

    py_csr_table.def("get_data", [](const dal::csr_table& table) -> py::object {
        return get_data_array(table);
    });

    py_csr_table.def("get_row_offsets", [](const dal::csr_table& table) -> py::object {
        auto iface = get_interface(table);
        auto array = iface->get_row_offsets();
        return py::cast(std::move(array));
    });

    py_csr_table.def("get_column_indices", [](const dal::csr_table& table) -> py::object {
        auto iface = get_interface(table);
        auto array = iface->get_column_indices();
        return py::cast(std::move(array));
    });

    instantiate_table_iface(py_csr_table);

    constexpr const supported_types_t* types = nullptr;
    instantiate_csr_constructor(py_csr_table, types);
}

} // namespace oneapi::dal::python::data_management
