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

#pragma once

#include <string>
#include <cstdint>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "oneapi/dal/detail/policy.hpp"

#include "onedal/common/device_lookup.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

#ifdef ONEDAL_DATA_PARALLEL

sycl::queue extract_queue(py::capsule capsule);
sycl::context extract_context(py::capsule capsule);
sycl::queue extract_from_capsule(py::capsule capsule);
sycl::queue get_queue_by_get_capsule(const py::object& syclobj);
sycl::queue get_queue_from_python(const py::object& syclobj);

using dp_policy_t = detail::data_parallel_policy;

dp_policy_t make_dp_policy(std::uint32_t id);
dp_policy_t make_dp_policy(const py::object& syclobj);
dp_policy_t make_dp_policy(const std::string& filter);
inline dp_policy_t make_dp_policy(const dp_policy_t& policy) {
    return dp_policy_t{ policy };
}

std::uint32_t get_device_id(const dp_policy_t& policy);
std::size_t get_used_memory(const py::object& syclobj);
std::string get_device_name(const dp_policy_t& policy);

/// TODO: This is a workaround class.
/// It hides deprecated ``sycl::ext::oneapi::filter_selector`` to get rid of build warnings
/// until a better solution is provided.
struct filter_selector_wrapper {
    filter_selector_wrapper(std::string Filter) : FilterSelector{Filter} {}

    int operator()(const sycl::device &Dev) {
        return FilterSelector(Dev);
    }

private:
    sycl::ext::oneapi::filter_selector FilterSelector;
};

#endif // ONEDAL_DATA_PARALLEL

template <typename Policy>
inline auto& instantiate_host_policy(py::class_<Policy>& policy) {
    policy.def(py::init<>());
    policy.def(py::init<Policy>());
    policy.def("get_device_id", [](const Policy&) -> std::uint32_t {
        return std::uint32_t{ 0u };
    });
    policy.def("get_device_name", [](const Policy&) -> std::string {
        return std::string{ "cpu" };
    });
    return policy;
}

} // namespace oneapi::dal::python
