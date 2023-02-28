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
#ifdef ONEDAL_DATA_PARALLEL
#include "onedal/common/pybind11_helpers.hpp"
#include <mpi.h>

namespace py = pybind11;

namespace oneapi::dal::python {

// TODO:
// Just for examples: will be removed.
class mpi_initializer
{
public:
    mpi_initializer() {}
    void init() { MPI_Init(NULL, NULL); }

    void fini() { MPI_Finalize(); }
};

ONEDAL_PY_INIT_MODULE(mpi_primitives) {
    //using mpi_initializer_t = mpi_initializer;
    py::class_<mpi_initializer>(m, "mpi_initializer")
        .def(py::init())
        .def("init", &mpi_initializer::init)
        .def("fini", &mpi_initializer::fini);
}
} // namespace oneapi::dal::python
#endif
