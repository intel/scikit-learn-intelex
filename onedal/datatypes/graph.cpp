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

#include <string>
#include <variant>
#define NO_IMPORT_ARRAY // import_array called in table.cpp
#include "onedal/datatypes/data_conversion.hpp"
#include "onedal/datatypes/numpy_helpers.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

template<typename Float>
void graph_constructor(py::module &m, const char* typestr) {
    py::class_<graph_t<Float>> graph_obj(m, typestr);
        graph_obj.def(py::init());
}

ONEDAL_PY_INIT_MODULE(graph) {
    //init_numpy();
    // init in table.cpp

    graph_constructor<float>(m, "graph_float");
    graph_constructor<double>(m, "graph_double");

    //py::class_<dal::preview::detail::topology> topo_obj(m, typestr);

    m.def("to_graph", [](py::object obj)-> graph_t<double> {
        auto* obj_ptr = obj.ptr();
        if (strcmp(Py_TYPE(obj_ptr)->tp_name, "csr_matrix") == 0 || strcmp(Py_TYPE(obj_ptr)->tp_name, "csr_array") == 0){
            PyObject *py_data = PyObject_GetAttrString(obj_ptr, "data");
            auto datatype = array_type(py_data);
            if (datatype == NPY_DOUBLE || datatype == NPY_CDOUBLE){
                return convert_to_undirected_graph<double>(obj_ptr, datatype);
            }
            //else if (datatype == NPY_FLOAT || datatype == NPY_CFLOAT || datatype == NPY_CFLOATLTR){
            //    return convert_to_undirected_graph<float>(obj_ptr, NPY_CFLOATLTR);
            //}
        }
        throw std::invalid_argument(
            "[convert_to_undirected_graph] Not available input format for converting Python object to onedal graph.");  
    });
}

} // namespace oneapi::dal::python
