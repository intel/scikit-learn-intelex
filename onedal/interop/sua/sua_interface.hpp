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

#include <array>
#include <cstdint>

#include <pybind11/pybind11.h>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#else // ONEDAL_DATA_PARALLEL
namespace sycl {
    class queue;
} // namespace sycl
#endif // ONEDAL_DATA_PARALLEL

#include "oneapi/dal/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::sua {

template <std::int64_t dim>
struct sua_interface {
    dal::data_type dtype;
    py::ssize_t offset = 0ul;
    std::shared_ptr<sycl::queue> queue;
    std::array<py::ssize_t, dim> shape;
    std::array<py::ssize_t, dim> strides;
    std::pair<std::uintptr_t, bool> data;
};

template <std::int64_t dim>
sua_interface<dim> get_sua_interface(const py::dict& sua);

template <std::int64_t dim>
py::dict get_sua_interface(const sua_interface<dim>& sua);

} // namespace oneapi::dal::python::interop::sua
