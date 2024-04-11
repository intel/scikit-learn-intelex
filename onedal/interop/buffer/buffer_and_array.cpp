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

#include "onedal/interop/buffer/common.hpp"
#include "onedal/interop/buffer/dtype_conversion.hpp"
#include "onedal/interop/buffer/buffer_and_array.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::buffer {

template <typename Type>
dal::array<Type> wrap_to_typed_array(py::buffer_info info) {
    (void)check_buffer<Type>(info);

    const std::int64_t casted_count = //
        detail::integral_cast<std::int64_t>(info.size);

    if (info.readonly) {
        const auto* ptr = reinterpret_cast<const Type*>(info.ptr);
        buf_deleter<const Type> deleter{ std::move(info) };
        auto cshared = std::shared_ptr<const Type>(ptr, std::move(deleter));
        return dal::array<Type>(std::move(cshared), casted_count);
    }
    else {
        auto* ptr = reinterpret_cast<Type*>(info.ptr);
        buf_deleter<Type> deleter{ std::move(info) };
        auto shared = std::shared_ptr<Type>(ptr, std::move(deleter));
        return dal::array<Type>(std::move(shared), casted_count);
    }
}

template <typename Type>
inline auto get_buffer_info(const dal::array<Type>& arr) {
    auto count = detail::integral_cast<py::ssize_t>(arr.get_count());
    constexpr auto size = static_cast<py::ssize_t>(sizeof(Type));
    constexpr auto one = static_cast<py::ssize_t>(1ul);
    auto* mut_ptr = const_cast<Type*>(arr.get_data());
    auto* raw_ptr = reinterpret_cast<void*>(mut_ptr);
    auto fmt = py::format_descriptor<Type>::format();
    auto immut = !arr.has_mutable_data();
    return py::buffer_info{ /*ptr=*/raw_ptr,
                            /*itemsize=*/size,
                            /*format=*/fmt,
                            /*ndim=*/one,
                            /*shape_in=*/{ count },
                            /*strides_in=*/{ size },
                            /*readonly=*/immut };
}

template <typename Type>
py::array_t<Type> wrap_from_typed_array(const dal::array<Type>& arr) {
    check_policy(arr);
    py::buffer_info info = get_buffer_info(arr);
    auto* const tmp_arr = new dal::array<Type>{ arr };
    auto capsule = py::capsule(tmp_arr, [](void* arr) -> void {
        delete reinterpret_cast<dal::array<Type>*>(arr);
    });
    return py::array_t<Type>(std::move(info), std::move(capsule));
}

py::object wrap_to_array(py::buffer_info info) {
    auto dt = convert_buffer_to_dal_type(info.format);
    auto wrap_buffer = [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        auto arr = wrap_to_typed_array<type_t>(std::move(info));
        return py::cast(new dal::array<type_t>(std::move(arr)));
    };

    return dal::detail::dispatch_by_data_type(dt, wrap_buffer);
}

py::object wrap_to_array(py::buffer buf) {
    py::buffer_info info = buf.request();
    return wrap_to_array(std::move(info));
}

void instantiate_wrap_to_array(py::module& pm) {
    pm.def("wrap_to_array", [](py::buffer buf) {
        return wrap_to_array(std::move(buf));
    });
}

template <typename... Types>
void instantiate_wrap_from_array_impl(py::module& pm, const std::tuple<Types...>* const = nullptr) {
    constexpr const char* name = "wrap_from_array";
    auto wrap = [&](auto type_tag) -> void {
        using type_t = std::decay_t<decltype(type_tag)>;
        using dal_array_t = dal::array<type_t>;
        using py_array_t = py::array_t<type_t>;
        pm.def(name, [](const dal_array_t& arr) -> py_array_t {
            return wrap_from_typed_array<type_t>(arr);
        });
    };
    return detail::apply(wrap, Types{}...);
}

void instantiate_wrap_from_array(py::module& pm) {
    constexpr const supported_types_t* types = nullptr;
    return instantiate_wrap_from_array_impl(pm, types);
}

void instantiate_buffer_and_array(py::module& pm) {
    instantiate_wrap_from_array(pm);
    instantiate_wrap_to_array(pm);
}

} // namespace oneapi::dal::python::interop::buffer
