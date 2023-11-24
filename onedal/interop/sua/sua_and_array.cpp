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

#include <optional>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/common.hpp"
#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/interop/common.hpp"
#include "onedal/interop/sua/sua_utils.hpp"
#include "onedal/interop/sua/sua_interface.hpp"
#include "onedal/interop/sua/sua_and_array.hpp"
#include "onedal/interop/sua/dtype_conversion.hpp"
#include "onedal/interop/utils/tensor_and_table.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::sua {

using interface_t = sua_interface<1l>;

template <typename Deleter>
inline py::object wrap_to_array(interface_t sua, Deleter&& del) {
    return utils::wrap_to_array(sua, std::forward<Deleter>(del));
}

py::object wrap_object_to_array(py::object object) {
    const py::dict dict = get_sua_interface(object);
    const interface_t sua = get_sua_interface<1l>(dict);
    const auto deleter = [object](auto* ptr) -> void {
        //check_correctness(raw_ptr, ptr);
        object.dec_ref();
    };
    auto result = wrap_to_array(std::move(sua), std::move(deleter));
    object.inc_ref();
    return result;
}

py::object wrap_dict_to_array(py::dict dict) {
    auto sua = get_sua_interface<1l>(dict);
    auto deleter = [](auto* ptr) -> void {
        //check_correctness(raw_ptr, ptr);
    };
    return wrap_to_array(std::move(sua), std::move(deleter));
}

void instantiate_wrap_to_array(py::module& pm) {
    constexpr const char name[] = "wrap_to_array";
    pm.def(name, &wrap_object_to_array);
    pm.def(name, &wrap_dict_to_array);
}

template <typename... Types>
inline void instantiate_wrap_from_array_impl(py::module& pm,
                                             const std::tuple<Types...>* const = nullptr) {
    constexpr const char name[] = "wrap_from_array";
    return detail::apply(
        [&](auto type_tag) -> void {
            using type_t = std::decay_t<decltype(type_tag)>;

            pm.def(name, [](const dal::array<type_t>& array) -> py::dict {
                using iface_t = sua_interface<1l>;
                auto interface = utils::wrap_from_array<iface_t>(array);
                return get_sua_interface(interface);
            });
        },
        Types{}...);
}

void instantiate_wrap_from_array(py::module& pm) {
    constexpr const supported_types_t* types = nullptr;
    return instantiate_wrap_from_array_impl(pm, types);
}

void instantiate_sua_and_array(py::module& pm) {
    instantiate_wrap_from_array(pm);
    instantiate_wrap_to_array(pm);
}

} // namespace oneapi::dal::python::interop::sua
