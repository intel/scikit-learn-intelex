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
#ifdef ONEDAL_DATA_PARALLEL_SPMD
#include "oneapi/dal/detail/spmd_policy.hpp"
#include "onedal/common/pybind11_helpers.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"
#include "dpctl4pybind11.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(spmd_policy) {
    import_dpctl();
    py::class_<dal::detail::spmd_policy<detail::data_parallel_policy>>(m, "spmd_data_parallel_policy")
        .def(py::init([](sycl::queue &q) {
            detail::data_parallel_policy local_policy = detail::data_parallel_policy(q);
            // TODO:
            // Communicator hardcoded. Implement passing spmd communicator.
            spmd::communicator<spmd::device_memory_access::usm> comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>(q);
            detail::spmd_policy<detail::data_parallel_policy> spmd_policy{ local_policy, comm };
            return spmd_policy;
        }));
}
} // namespace oneapi::dal::python
#endif
