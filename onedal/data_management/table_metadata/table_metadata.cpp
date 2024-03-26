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

#include "oneapi/dal/table/common.hpp"

#include "onedal/common.hpp"
#include "onedal/common/serialization.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::data_management {

void instantiate_table_metadata(py::module& pm) {
    constexpr const char name[] = "table_metadata";
    using dtype_arr_t = dal::array<dal::data_type>;
    using ftype_arr_t = dal::array<dal::feature_type>;

    py::class_<dal::table_metadata> py_metadata(pm, name);
    py_metadata.def(py::init());
    py_metadata.def(py::init<dal::table_metadata>());
    py_metadata.def(py::init<dtype_arr_t, ftype_arr_t>());
    py_metadata.def(py::pickle(
        [](const table_metadata& m) -> py::bytes {
            return serialize(m);
        },
        [](const py::bytes& bytes) -> table_metadata {
            return deserialize<table_metadata>(bytes);
        }));
    py_metadata.def("get_data_type", &table_metadata::get_data_type);
    py_metadata.def("get_data_types", &table_metadata::get_data_types);
    py_metadata.def("get_feature_type", &table_metadata::get_feature_type);
    py_metadata.def("get_feature_types", &table_metadata::get_feature_types);
    py_metadata.def("get_feature_count", &table_metadata::get_feature_count);
}

} // namespace oneapi::dal::python::data_management
