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

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "onedal/common/policy_common.hpp"

namespace oneapi::dal::python {

#ifdef ONEDAL_DATA_PARALLEL

constexpr const char unknown_device[] = "Unknown device";
constexpr const char py_capsule_name[] = "PyCapsule";
constexpr const char get_capsule_name[] = "_get_capsule";
constexpr const char queue_capsule_name[] = "SyclQueueRef";
constexpr const char context_capsule_name[] = "SyclContextRef";

sycl::queue extract_queue(py::capsule capsule) {
    constexpr const char* gtr_name = queue_capsule_name;
    constexpr std::size_t gtr_size = sizeof(queue_capsule_name);
    if (std::strncmp(capsule.name(), gtr_name, gtr_size) != 0) {
        throw std::runtime_error("Capsule should contain \"SyclQueueRef\"");
    }
    return sycl::queue{ *capsule.get_pointer<sycl::queue>() };
}

sycl::context extract_context(py::capsule capsule) {
    constexpr const char* gtr_name = context_capsule_name;
    constexpr std::size_t gtr_size = sizeof(context_capsule_name);
    if (std::strncmp(capsule.name(), gtr_name, gtr_size) != 0) {
        throw std::runtime_error("Capsule should contain \"SyclContextRef\"");
    }
    return sycl::context{ *capsule.get_pointer<sycl::context>() };
}

sycl::queue extract_from_capsule(py::capsule capsule) {
    const char* const name = capsule.name();
    if (std::strncmp(name, context_capsule_name, sizeof(context_capsule_name)) == 0) {
        const auto ctx = extract_context(std::move(capsule));
        return sycl::queue{ ctx, sycl::default_selector_v };
    }
    else if (std::strncmp(name, queue_capsule_name, sizeof(queue_capsule_name)) == 0) {
        return extract_queue(std::move(capsule));
    }
    else {
        throw std::runtime_error("Capsule should contain \"SyclQueueRef\" or \"SyclContextRef\"");
    }
}

sycl::queue get_queue_by_get_capsule(const py::object& syclobj) {
    auto attr = syclobj.attr(get_capsule_name);
    auto capsule = attr().cast<py::capsule>();
    return extract_from_capsule(std::move(capsule));
}

sycl::queue get_queue_from_python(const py::object& syclobj) {
    static auto pycapsule = py::cast(py_capsule_name);
    if (py::hasattr(syclobj, get_capsule_name)) {
        return get_queue_by_get_capsule(syclobj);
    }
    else if (py::isinstance(syclobj, pycapsule)) {
        const auto caps = syclobj.cast<py::capsule>();
        return extract_from_capsule(std::move(caps));
    }
    else {
        throw std::runtime_error("Unable to interpret \"syclobj\"");
    }
}

sycl::queue get_queue_by_filter_string(const std::string& filter) {
    sycl::ext::oneapi::filter_selector selector{ filter };
    return sycl::queue{ selector };
}

sycl::queue get_queue_by_device_id(std::uint32_t id) {
    if (auto device = get_device_by_id(id)) {
        return sycl::queue{ device.value() };
    }
    else {
        throw std::runtime_error(unknown_device);
    }
}

std::string get_device_name(const sycl::queue& queue) {
    const auto& device = queue.get_device();
    if (device.is_gpu()) {
        return { "gpu" };
    }
    else if (device.is_cpu()) {
        return { "cpu" };
    }
    else {
        return { "unknown" };
    }
}

std::uint32_t get_device_id(const sycl::queue& queue) {
    const auto& device = queue.get_device();
    if (auto id = get_device_id(device)) {
        return std::uint32_t{ id.value() };
    }
    else {
        throw std::runtime_error(unknown_device);
    }
}

dp_policy_t make_dp_policy(std::uint32_t id) {
    sycl::queue queue = get_queue_by_device_id(id);
    return dp_policy_t{ std::move(queue) };
}

dp_policy_t make_dp_policy(const py::object& syclobj) {
    sycl::queue queue = get_queue_from_python(syclobj);
    return dp_policy_t{ std::move(queue) };
}

dp_policy_t make_dp_policy(const std::string& filter) {
    sycl::queue queue = get_queue_by_filter_string(filter);
    return dp_policy_t{ std::move(queue) };
}

std::uint32_t get_device_id(const dp_policy_t& policy) {
    const auto& queue = policy.get_queue();
    return get_device_id(queue);
}

std::string get_device_name(const dp_policy_t& policy) {
    const auto& queue = policy.get_queue();
    return get_device_name(queue);
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::python
