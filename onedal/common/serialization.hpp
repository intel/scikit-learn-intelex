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

#pragma once

#include <pybind11/pybind11.h>

#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/archives.hpp"

namespace oneapi::dal::python {

template <typename T>
pybind11::bytes serialize(const T& original) {
    detail::binary_output_archive archive;
    detail::serialize(original, archive);
    const auto data = archive.to_array();
    return { reinterpret_cast<const char*>(data.get_data()),
             dal::detail::integral_cast<std::size_t>(archive.get_size()) };
}

template <typename T>
T deserialize(const pybind11::bytes& bytes) {
    T deserialized;
    const std::string str = bytes;

    detail::binary_input_archive archive{ reinterpret_cast<const byte_t*>(str.c_str()),
                                          dal::detail::integral_cast<std::int64_t>(str.size()) };
    detail::deserialize(deserialized, archive);
    return deserialized;
}

} // namespace oneapi::dal::python
