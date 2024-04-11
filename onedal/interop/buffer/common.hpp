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

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "oneapi/dal/array.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::buffer {

template <typename Type>
inline void check_buffer(const py::buffer_info& info, 
                            std::size_t ndim = 1ul) {
    const auto fmt = py::format_descriptor<Type>::format();
    if (info.format != fmt) {
        throw std::domain_error("Wrong type");
    }
    if (info.itemsize != sizeof(Type)) {
        throw std::range_error("Wrong type size");
    }
    if (info.ndim != ndim) {
        throw std::range_error("Wrong dimensionality");
    }
}

template <typename Type, std::size_t dim = 1ul>
class buf_deleter {
public:
    buf_deleter() = default;
    buf_deleter(buf_deleter&&) = default;
    buf_deleter(const buf_deleter&) = default;
    buf_deleter(py::buffer_info info) : info_{ std::move(info) } {}

    void operator() (Type* const ptr) {
        check_buffer<Type>(info_, dim);
        py::buffer_info empty_info_;
        std::swap(info_, empty_info_);
    }

private:
    py::buffer_info info_;
};

template <typename Type>
inline void check_policy(const dal::array<Type>& array) {
#ifdef ONEDAL_DATA_PARALLEL
    constexpr char err[] = "Not CPU devices are not supported";
    if (auto queue = array.get_queue()) {
        const auto& device = queue.value().get_device();
        if (!device.is_cpu()) throw std::runtime_error(err);
    }
#endif // ONEDAL_DATA_PARALLEL
} 

} // namespace oneapi::dal::python::interop::buffer
