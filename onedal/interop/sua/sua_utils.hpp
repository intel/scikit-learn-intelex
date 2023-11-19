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

py::dict get_sua_interface(const py::object& obj);

bool is_sua_readonly(const py::dict& sua);
py::tuple get_sua_data(const py::dict& sua);
py::tuple get_sua_shape(const py::dict& sua);
std::int64_t get_sua_ndim(const py::dict& sua);
std::int64_t get_sua_count(const py::dict& sua);
std::uintptr_t get_sua_ptr(const py::dict& sua);
dal::data_type get_sua_dtype(const py::dict& sua);

#ifdef ONEDAL_DATA_PARALLEL
sycl::queue extract_queue(py::capsule capsule);
sycl::queue get_sua_queue(const py::dict& sua);
#endif // ONEDAL_DATA_PARALLEL

py::capsule pack_queue(const std::shared_ptr<sycl::queue>& queue);
std::shared_ptr<sycl::queue> get_sua_shared_queue(const py::dict& sua);

} // namespace oneapi::dal::python::interop::sua
