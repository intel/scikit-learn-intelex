/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/csr.hpp"

#include "onedal/datatypes/data_conversion.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(table) {
    py::class_<table>(m, "table")
        .def(py::init())
        .def_property_readonly("has_data", &table::has_data)
        .def_property_readonly("column_count", &table::get_column_count)
        .def_property_readonly("row_count", &table::get_row_count)
        .def_property_readonly("kind", [](const table& t) {
            if (t.get_kind() == 0) { // TODO: expose empty table kind
                return "empty";
            }
            if (t.get_kind() == homogen_table::kind()) {
                return "homogen";
            }
            if (t.get_kind() == detail::csr_table::kind()) {
                return "csr";
            }
            return "unknown";
        });

    m.def("from_numpy", [](py::object obj) {
        auto* obj_ptr = obj.ptr();
        return convert_to_table(obj_ptr);
    });

    m.def("to_numpy", [](dal::table& t) -> py::handle {
        auto* obj_ptr = convert_to_numpy(t);
        return obj_ptr;
    });
}

} // namespace oneapi::dal::python
