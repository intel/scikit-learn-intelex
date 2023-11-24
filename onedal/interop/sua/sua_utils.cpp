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

#include <cstring>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/common/policy_common.hpp"
#include "onedal/interop/common.hpp"
#include "onedal/interop/sua/sua_utils.hpp"
#include "onedal/interop/sua/sua_interface.hpp"
#include "onedal/interop/sua/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::sua {

py::dict get_sua_interface(const py::object& obj) {
    constexpr const char name[] = "__sycl_usm_array_interface__";
    return obj.attr(name).cast<py::dict>();
}

py::tuple get_sua_data(const py::dict& sua) {
    py::tuple result = sua["data"].cast<py::tuple>();
    if (result.size() != py::ssize_t{ 2ul }) {
        throw std::length_error("Size of \"data\" tuple should be 2");
    }
    return result;
}

std::uintptr_t get_sua_ptr(const py::dict& sua) {
    const py::tuple data = get_sua_data(sua);
    return data[0ul].cast<std::uintptr_t>();
}

bool is_sua_readonly(const py::dict& sua) {
    const py::tuple data = get_sua_data(sua);
    return data[1ul].cast<bool>();
}

py::tuple get_sua_shape(const py::dict& sua) {
    py::tuple shape = sua["shape"].cast<py::tuple>();
    if (shape.size() == py::ssize_t{ 0ul }) {
        throw std::runtime_error("Wrong number of dimensions");
    }
    return shape;
}

std::int64_t get_sua_ndim(const py::dict& sua) {
    py::tuple shape = get_sua_shape(sua);
    const py::ssize_t raw_ndim = shape.size();
    return detail::integral_cast<std::int64_t>(raw_ndim);
}

std::int64_t get_sua_count(const py::dict& sua) {
    py::tuple shape = get_sua_shape(sua);

    py::ssize_t result = 1ul;
    const auto sentinel = shape.end();
    for (auto it = shape.begin(); it != sentinel; ++it) {
        const py::ssize_t dimension = it->cast<py::size_t>();
        result = detail::check_mul_overflow(result, dimension);
    }
    return detail::integral_cast<std::int64_t>(result);
}

dal::data_type get_sua_dtype(const py::dict& sua) {
    auto dtype = sua["typestr"].cast<std::string>();
    return convert_sua_to_dal_type(std::move(dtype));
}

#ifdef ONEDAL_DATA_PARALLEL

sycl::queue get_sua_queue(const py::dict& sua) {
    auto syclobj = sua["syclobj"].cast<py::object>();
    return get_queue_from_python(syclobj);
}

std::shared_ptr<sycl::queue> get_sua_shared_queue(const py::dict& sua) {
    sycl::queue queue = get_sua_queue(sua);
    return std::make_shared<sycl::queue>(std::move(queue));
}

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

#else // ONEDAL_DATA_PARALLEL

std::shared_ptr<sycl::queue> get_sua_shared_queue(const py::dict& sua) {
    return nullptr;
}

py::capsule pack_queue(const std::shared_ptr<sycl::queue>& queue) {
    throw std::runtime_error("SYCL us not supported");
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::python::interop::sua
