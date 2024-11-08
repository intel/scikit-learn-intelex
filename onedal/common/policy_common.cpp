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

#include <Python.h>
#include <pybind11/pybind11.h>

#include "onedal/common/policy_common.hpp"

namespace oneapi::dal::python {

#ifdef ONEDAL_DATA_PARALLEL

constexpr const char unknown_device[] = "Unknown device";
constexpr const char get_capsule_name[] = "_get_capsule";
constexpr const char queue_capsule_name[] = "SyclQueueRef";
constexpr const char context_capsule_name[] = "SyclContextRef";
constexpr const char device_name[] = "sycl_device";
constexpr const char get_filter_name[] = "get_filter_string";

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

sycl::queue get_queue_by_pylong_pointer(const py::int_& syclobj) {
    // 
    void *ptr = PyLong_AsVoidPtr(syclobj.ptr());
    // assumes that the PyLong is a pointer to a queue
    return static_cast<sycl::queue>(*ptr);
}

sycl::queue get_queue_by_filter_string(const std::string& filter) {
    filter_selector_wrapper selector{ filter };
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

sycl::queue get_queue_from_python(const py::object& syclobj) {
    if (py::hasattr(syclobj, get_capsule_name)) {
        return get_queue_by_get_capsule(syclobj);
    }
    else if (py::isinstance<py::int_>(syclobj)) {
        return get_queue_by_pylong_pointer(syclobj);
    }
    else if (py::isinstance<py::capsule>(syclobj)) {
        const auto caps = syclobj.cast<py::capsule>();
        return extract_from_capsule(std::move(caps));
    }
    else if (py::hasattr(syclobj, device_name) && py::hasattr(syclobj.attr(device_name), get_filter_name)) {
        auto attr = syclobj.attr(device_name).attr(get_filter_name);
        return get_queue_by_filter_string(attr().cast<std::string>());
    }
    else
    {
        throw std::runtime_error("Unable to interpret \"syclobj\"");
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

std::size_t get_used_memory(const py::object& syclobj){
    const auto& device =  get_queue_from_python(syclobj).get_device();
    std::size_t total_memory = device.get_info<sycl::info::device::global_mem_size>();
    std::size_t free_memory = device.get_info<sycl::ext::intel::info::device::free_memory>();
    return total_memory - free_memory;
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

// Create `SyclQueueRef` PyCapsule that represents an opaque value of
// sycl::queue.
py::capsule pack_queue(const std::shared_ptr<sycl::queue>& queue) {
    static const char queue_capsule_name[] = "SyclQueueRef";
    if (queue.get() == nullptr) {
        throw std::runtime_error("Empty queue");
    }
    else {
        void (*deleter)(void*) = [](void* const queue) -> void {
            delete reinterpret_cast<sycl::queue* const>(queue);
        };

        sycl::queue* ptr = new sycl::queue{ *queue };
        void* const raw = reinterpret_cast<void*>(ptr);

        py::capsule capsule(raw, deleter);
        capsule.set_name(queue_capsule_name);
        return capsule;
    }
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::python
