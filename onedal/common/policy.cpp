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

#include "oneapi/dal/detail/policy.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(policy) {
    py::class_<detail::host_policy>(m, "host_policy").def(py::init());

#ifdef ONEDAL_DATA_PARALLEL
    py::class_<detail::data_parallel_policy>(m, "data_parallel_policy")
        .def(py::init([](const std::string& device_type) {
            if (device_type == "gpu") {
                return new detail::data_parallel_policy(sycl::gpu_selector());
            }

            return new detail::data_parallel_policy(sycl::cpu_selector());
        }));
#endif
}

} // namespace oneapi::dal::python
