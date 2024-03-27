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
#include "onedal/common/policy_common.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

using host_policy_t = dal::detail::host_policy;
using default_host_policy_t = dal::detail::default_host_policy;

void instantiate_host_policy(py::module& m) {
    constexpr const char name[] = "host_policy";
    py::class_<host_policy_t> policy(m, name);
    policy.def(py::init<host_policy_t>());
    instantiate_host_policy(policy);
}

void instantiate_default_host_policy(py::module& m) {
    constexpr const char name[] = "default_host_policy";
    py::class_<default_host_policy_t> policy(m, name);
    policy.def(py::init<default_host_policy_t>());
    instantiate_host_policy(policy);
}

#ifdef ONEDAL_DATA_PARALLEL

using data_parallel_policy_t = dal::detail::data_parallel_policy;

void instantiate_data_parallel_policy(py::module& m) {
    constexpr const char name[] = "data_parallel_policy";
    py::class_<data_parallel_policy_t> policy(m, name);
    policy.def(py::init<data_parallel_policy_t>());
    policy.def(py::init([](std::uint32_t id) {
        return make_dp_policy(id);
    }));
    policy.def(py::init([](const std::string& filter) {
        return make_dp_policy(filter);
    }));
    policy.def(py::init([](const py::object& syclobj) {
        return make_dp_policy(syclobj);
    }));
    policy.def("get_device_id", [](const data_parallel_policy_t& policy) {
        return get_device_id(policy);
    });
    policy.def("get_device_name", [](const data_parallel_policy_t& policy) {
        return get_device_name(policy);
    });
}

#endif // ONEDAL_DATA_PARALLEL

ONEDAL_PY_INIT_MODULE(policy) {
    instantiate_host_policy(m);
    instantiate_default_host_policy(m);
#ifdef ONEDAL_DATA_PARALLEL
    instantiate_data_parallel_policy(m);
#endif // ONEDAL_DATA_PARALLEL
}

} // namespace oneapi::dal::python
