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

#include <unordered_map>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/interop/common.hpp"
#include "onedal/interop/buffer/dtype_conversion.hpp"

namespace oneapi::dal::python::interop::buffer {

using fwd_map_t = std::unordered_map<std::string, dal::data_type>;
using inv_map_t = std::unordered_map<dal::data_type, std::string>;

template <typename... Types>
inline auto make_fwd_map(const std::tuple<Types...>* const = nullptr) {
    fwd_map_t result(sizeof...(Types));

    dal::detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;
            constexpr auto dal_v = detail::make_data_type<type_t>();
            auto buf_v = py::format_descriptor<type_t>::format();
            result.emplace(buf_v, dal_v);
        },
        Types{}...);

    return result;
}

static const fwd_map_t& get_fwd_map() {
    constexpr supported_types_t* const types = nullptr;
    static const fwd_map_t body = make_fwd_map(types);
    return body;
}

static const inv_map_t& get_inv_map() {
    static const auto body = inverse_map(get_fwd_map());
    return body;
}

dal::data_type convert_buffer_to_dal_type(std::string dtype) {
    return get_fwd_map().at(dtype);
}

std::string convert_dal_to_buffer_type(dal::data_type dtype) {
    return get_inv_map().at(dtype);
}

void instantiate_buffer_dtype_convert(py::module& pm) {
    instantiate_convert_dal_to_buffer_dtype(pm);
    instantiate_convert_buffer_to_dal_dtype(pm);
}

void instantiate_convert_dal_to_buffer_dtype(py::module& pm) {
    constexpr const char name[] = "convert_from_dal_dtype";
    pm.def(name, [](dal::data_type dt) -> std::string {
        return convert_dal_to_buffer_type(dt);
    });
    pm.def(name, [](std::int64_t dt) -> std::string {
        auto casted = static_cast<dal::data_type>(dt);
        return convert_dal_to_buffer_type(casted);
    });
}

void instantiate_convert_buffer_to_dal_dtype(py::module& pm) {
    constexpr const char name[] = "convert_to_dal_dtype";
    pm.def(name, [](std::string dt) -> dal::data_type {
        return convert_buffer_to_dal_type(std::move(dt));
    });
}

} // namespace oneapi::dal::python::interop::buffer
