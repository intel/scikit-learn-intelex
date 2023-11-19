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

#include <variant>
#include <stdexcept>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/array_utils.hpp"

#include "onedal/common.hpp"

#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/data_management/common.hpp"
#include "onedal/data_management/array/array.hpp"

namespace oneapi::dal::python::data_management {

template <typename Type>
inline std::string name_array() {
    auto type = py::format_descriptor<Type>::format();
    return std::string("array_") + std::string(type);
}

template <typename Type, bool need_mutable = false>
inline void check_access(const dal::array<Type>& arr, std::int64_t idx) {
    if (idx < 0l || arr.get_count() < idx) {
        throw std::out_of_range("Out of range access to array");
    }
    if (need_mutable && !arr.has_mutable_data()) {
        throw std::domain_error("Mutable access to immutable array");
    }
}

template <typename T>
inline auto get_policy(const dal::array<T>& arr) {
    return dal::detail::get_impl(arr).get_policy();
}

template <typename Policy>
constexpr inline bool is_host_policy_v = dal::detail::is_one_of_v<Policy, //
                                                                  detail::default_host_policy,
                                                                  detail::host_policy>;

template <typename InpPolicy, typename OutPolicy>
inline bool need_copy(const InpPolicy& inp, const OutPolicy& out) {
    using out_policy_t = std::decay_t<decltype(inp)>;
    using inp_policy_t = std::decay_t<decltype(out)>;
    constexpr bool is_inp_host = is_host_policy_v<inp_policy_t>;
    constexpr bool is_out_host = is_host_policy_v<out_policy_t>;
    constexpr bool result = !(is_inp_host && is_out_host);
    return result;
}

// TODO: Check for the same policy
template <typename Policy, typename T>
inline dal::array<T> to_policy(const Policy& out, const dal::array<T>& source) {
    return std::visit(
        [&](const auto& inp) -> dal::array<T> {
            if (need_copy(inp, out)) {
                return detail::copy(out, source);
            }
            else {
                return dal::array<T>{ source };
            }
        },
        get_policy(source));
}

template <typename Policy, typename Array>
inline void instantiate_to_policy(py::class_<Array>& py_array) {
    constexpr const char name[] = "to_policy";
    py_array.def(name, [](const Array& source, const Policy& policy) {
        auto result = to_policy(policy, source);
        return py::cast(std::move(result));
    });
}

template <typename Type>
void instantiate_array_by_type(py::module& pm) {
    const auto name = name_array<Type>();
    const char* const c_name = name.c_str();

    using array_t = oneapi::dal::array<Type>;

    py::class_<array_t> py_array(pm, c_name);
    py_array.def(py::init<>());
    py_array.def(py::init<array_t>());
    py_array.def(py::pickle(
        [](const array_t& m) -> py::bytes {
            return serialize(m);
        },
        [](const py::bytes& bytes) -> array_t {
            return deserialize<array_t>(bytes);
        }));
    py_array.def("__len__", &array_t::get_count);
    py_array.def("get_count", &array_t::get_count);
    py_array.def("has_mutable_data", &array_t::has_mutable_data);
    py_array.def("has_data", [](const array_t& array) -> bool {
        return array.get_count() > std::int64_t(0l);
    });
    py_array.def("get_slice",
                 [](const array_t& array, std::int64_t first, std::int64_t last) -> array_t {
                     constexpr std::int64_t zero = 0l;
                     const range<std::int64_t> outer{ zero, array.get_count() };
                     const range<std::int64_t> inner{ first, last };
                     check_in_range<std::int64_t>(inner, outer);
                     return array.get_slice(first, last);
                 });
    py_array.def("get_data", [](const array_t& array) -> std::uintptr_t {
        const Type* const raw = array.get_data();
        return reinterpret_cast<std::uintptr_t>(raw);
    });
    py_array.def("get_size_in_bytes", [](const array_t& array) -> std::int64_t {
        using oneapi::dal::detail::integral_cast;
        using oneapi::dal::detail::check_mul_overflow;
        const auto size = integral_cast<std::int64_t>(sizeof(Type));
        return check_mul_overflow<std::int64_t>(array.get_count(), size);
    });
    py_array.def("get_dtype", [](const array_t&) -> dal::data_type {
        constexpr auto dtype = dal::detail::make_data_type<Type>();
        return dtype;
    });
    py_array.def("get_policy", [](const array_t& arr) -> py::object {
        return std::visit(
            [](const auto& policy) -> py::object {
                return py::cast(policy);
            },
            get_policy(arr));
    });
    py_array.def("__getitem__", [](const array_t& arr, std::int64_t idx) -> Type {
        check_access<Type, false>(arr, idx);
        return *(arr.get_data() + idx);
    });
    py_array.def("__setitem__", [](const array_t& arr, std::int64_t idx, Type val) {
        check_access<Type, true>(arr, idx);
        *(arr.get_mutable_data() + idx) = val;
    });
    py_array.def_property_readonly("__is_onedal_array__", [](const array_t&) -> bool {
        return true;
    });
    instantiate_to_policy<detail::host_policy>(py_array);
    instantiate_to_policy<detail::default_host_policy>(py_array);
#ifdef ONEDAL_DATA_PARALLEL
    instantiate_to_policy<detail::data_parallel_policy>(py_array);
#endif // ONEDAL_DATA_PARALLEL
}

template <typename... Types>
inline void instantiate_array_impl(py::module& pm, const std::tuple<Types...>* const = nullptr) {
    auto instantiate = [&](auto type_tag) -> void {
        using type_t = std::decay_t<decltype(type_tag)>;
        return instantiate_array_by_type<type_t>(pm);
    };
    return detail::apply(instantiate, Types{}...);
}

template <typename... Types>
inline void instantiate_make_array(py::module& pm, const std::tuple<Types...>* const = nullptr) {
    constexpr const char name[] = "make_array";
    auto instantiate = [&](auto type_tag) -> void {
        using type_t = std::decay_t<decltype(type_tag)>;
        using array_t = dal::array<type_t>;
        pm.def(name, [](const array_t& arr) -> array_t {
            return array_t{ arr };
        });
    };
    return detail::apply(instantiate, Types{}...);
}

void instantiate_array(py::module& pm) {
    constexpr const supported_types_t* types = nullptr;
    (void)instantiate_array_impl(pm, types);
    (void)instantiate_make_array(pm, types);
}

} // namespace oneapi::dal::python::data_management
