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

#include <pybind11/pybind11.h>
#include <sstream>
#include "onedal/common.hpp"

#include "onedal/data_management/table/table.hpp"
#include "onedal/data_management/table/table_iface.hpp"

#include "oneapi/dal/common.hpp"

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/heterogen.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::data_management {

void instantiate_data_layout(py::module& pm) {
    py::enum_<data_layout> py_data_layout(pm, "data_layout");
    py_data_layout.value("unknown", data_layout::unknown);
    py_data_layout.value("row_major", data_layout::row_major);
    py_data_layout.value("column_major", data_layout::column_major);
    py_data_layout.export_values();
}

void instantiate_table_kind(py::module& pm) {
    py::enum_<table_kind> py_table_kind(pm, "table_kind");
    auto define_kind = [&](auto name, std::int64_t value) -> void {
        py_table_kind.value(name, static_cast<table_kind>(value));
    };
    define_kind("empty", table{}.get_kind());
    define_kind("csr", csr_table::kind());
    define_kind("homogen", homogen_table::kind());
    define_kind("heterogen", heterogen_table::kind());
    py_table_kind.export_values();
}

void instantiate_table_enums(py::module& pm) {
    instantiate_data_layout(pm);
    instantiate_table_kind(pm);
}

void instantiate_table(py::module& pm) {
    constexpr char name[] = "table";
    py::class_<dal::table> py_table(pm, name);

    py_table.def(py::init<dal::table>());
    py_table.def(py::init<dal::csr_table>());
    py_table.def(py::init<dal::homogen_table>());
    py_table.def(py::init<dal::heterogen_table>());

    return instantiate_table_iface(py_table);
}

} // namespace oneapi::dal::python::data_management
