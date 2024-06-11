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

#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"


#include "onedal/datatypes/data_conversion.hpp"
#include "onedal/datatypes/numpy_helpers.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

static void* init_numpy() {
    import_array();
    return nullptr;
}

ONEDAL_PY_INIT_MODULE(graph) {
    init_numpy();

    template<typename Float>
    void graph_constructor(py::module &m, const char* typestr) {
        using graph_t = dal::preview::undirected_adjacency_vector_graph<std::int32_t, Float>;
        py::class_<graph_t> graph_obj(m, typestr)
            .def(py::init());
    }

    graph_constructor<float>(m, "graph_float")
    graph_constructor<float>(m, "graph_double")


    m.def("to_graph", [](py::object obj) {
        auto* obj_ptr = obj.ptr();
        return convert_to_undirected_graph(obj_ptr);
    });
}

} // namespace oneapi::dal::python
