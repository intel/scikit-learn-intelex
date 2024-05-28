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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/interop/buffer/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::buffer {

template <typename Type>
void check_buffer(const py::buffer_info& info, std::size_t ndim = 1ul) {
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

} // namespace oneapi::dal::python::interop::buffer
