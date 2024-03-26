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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/common.hpp"

#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/data_management/common.hpp"
#include "onedal/data_management/chunked_array/chunked_array.hpp"

namespace oneapi::dal::python::data_management {

template <typename Type>
inline std::string name_chunked_array() {
    auto type = py::format_descriptor<Type>::format();
    return std::string("chunked_array_") + type;
}

template <typename Type>
void instantiate_chunked_array_by_type(py::module& m) {
    using array_t = dal::array<Type>;
    using chunked_array_t = dal::chunked_array<Type>;
    const auto name = name_chunked_array<Type>();
    const char* c_name = name.c_str();

    py::class_<chunked_array_t> py_array(m, c_name);
    py_array.def(py::init<>());
    py_array.def(py::pickle(
        [](const chunked_array_t& m) -> py::bytes {
            return serialize(m);
        },
        [](const py::bytes& bytes) -> chunked_array_t {
            return deserialize<chunked_array_t>(bytes);
        }));
    py_array.def(py::init<std::int64_t>());
    py_array.def("validate", &chunked_array_t::validate);
    py_array.def("get_count", &chunked_array_t::get_count);
    py_array.def("is_contiguous", &chunked_array_t::is_contiguous);
    py_array.def("get_chunk_count", &chunked_array_t::get_chunk_count);
    py_array.def("get_size_in_bytes", &chunked_array_t::get_size_in_bytes);
    py_array.def("have_same_policies", &chunked_array_t::have_same_policies);
    py_array.def("flatten", [](const chunked_array_t& chunked) -> array_t {
        return chunked.flatten();
    });
    py_array.def("get_mut_chunk", [](chunked_array_t& chunked, std::int64_t chunk) -> array_t {
        return chunked.get_chunk(chunk);
    });
    py_array.def("get_chunk", [](const chunked_array_t& chunked, std::int64_t chunk) -> array_t {
        return chunked.get_chunk(chunk);
    });
    py_array.def("get_slice",
                 [](const chunked_array_t& chunked,
                    std::int64_t first,
                    std::int64_t last) -> chunked_array_t {
                     constexpr std::int64_t zero = 0l;
                     const range<std::int64_t> outer{ zero, chunked.get_count() };
                     const range<std::int64_t> inner{ first, last };
                     check_in_range<std::int64_t>(inner, outer);
                     return chunked.get_slice(first, last);
                 });
    py_array.def("set_chunk",
                 [](chunked_array_t& chunked, std::int64_t chunk, const array_t& array) {
                     chunked.set_chunk(chunk, array);
                 });
    py_array.def("get_dtype", [](const chunked_array_t&) -> dal::data_type {
        constexpr auto dtype = detail::make_data_type<Type>();
        return dtype;
    });
}

template <typename... Types>
inline void instantiate_chunked_array_impl(py::module& pm,
                                           const std::tuple<Types...>* const = nullptr) {
    auto instantiate = [&](auto type_tag) -> void {
        using type_t = std::decay_t<decltype(type_tag)>;
        return instantiate_chunked_array_by_type<type_t>(pm);
    };
    return detail::apply(instantiate, Types{}...);
}

void instantiate_chunked_array(py::module& pm) {
    constexpr const supported_types_t* types = nullptr;
    return instantiate_chunked_array_impl(pm, types);
}

} // namespace oneapi::dal::python::data_management
