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

#pragma once

#include <string>
#include <cstdint>
#include <sstream>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::data_management {

// Opaque enum4 declaration according to:
// https://en.cppreference.com/w/cpp/language/enum
enum table_kind : std::int64_t;

template <typename Table>
inline void instantiate_table_iface(py::class_<Table>& py_table) {
    py_table.def(py::pickle(
        [](const Table& m) -> py::bytes {
            return serialize(m);
        },
        [](const py::bytes& bytes) -> Table {
            return deserialize<Table>(bytes);
        })
    );
    py_table.def(py::init<>());
    py_table.def(py::init<Table>());
    py_table.def("get_metadata", &Table::get_metadata);
    py_table.def("get_row_count", &Table::get_row_count);
    py_table.def("get_data_layout", &Table::get_data_layout);
    py_table.def("get_column_count", &Table::get_column_count);
    py_table.def("has_data", [](const Table& t) -> bool {
        return t.has_data();
    });
    py_table.def("get_kind", [](const Table& t) -> table_kind {
        return static_cast<table_kind>(t.get_kind());
    });
    py_table.def_property_readonly("__is_onedal_table__", [](const Table&) -> bool {
        return true;
    });

    py_table.def("__repr__", [](const Table& t) {
        std::stringstream stream;

        static const auto type = py::type::handle_of<Table>();
        static const auto name = std::string{ py::str(type) };
        stream << "<oneDAL table of kind: " << t.get_kind();
        stream << " packed into " << name << '>';
        return stream.str();
    });
}

} // namespace oneapi::dal::python::data_management
