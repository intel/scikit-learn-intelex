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

#include <string>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::buffer {

dal::data_type convert_buffer_to_dal_type(std::string dtype);
std::string convert_dal_to_buffer_type(dal::data_type dtype);

void instantiate_buffer_dtype_convert(py::module& pm);
void instantiate_convert_dal_to_buffer_dtype(py::module& pm);
void instantiate_convert_buffer_to_dal_dtype(py::module& pm);

} // namespace oneapi::dal::python::interop::buffer
