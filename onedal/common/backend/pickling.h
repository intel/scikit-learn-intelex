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

#include "oneapi/dal/detail/serialization.hpp"
#include "Python.h"

namespace oneapi::dal::python {

template <typename T>
PyObject* serialize_si(T* original) {
    detail::binary_output_archive archive;
    detail::serialize(*original, archive);
    const byte_t* data = archive.get_data();
    const Py_ssize_t buf_len = archive.get_size();
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(data), buf_len);
}

template <typename T>
T* deserialize_si(PyObject* py_bytes) {
    T deserialized;
    char* buf;
    Py_ssize_t buf_len;
    PyBytes_AsStringAndSize(py_bytes, &buf, &buf_len);
    detail::binary_input_archive archive{ reinterpret_cast<byte_t*>(buf), buf_len };
    detail::deserialize(deserialized, archive);
    return new T(deserialized);
}

} // namespace oneapi::dal::python
