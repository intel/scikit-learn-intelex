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

#include "onedal/common/type_utils.hpp"

#define ONEDAL_PY_DECLARE_INSTANTIATOR(func_name)                \
    struct instantiator_##func_name {                            \
        instantiator_##func_name(pybind11::module_& m) : m(m) {} \
                                                                 \
        template <typename... Args>                              \
        constexpr void run() {                                   \
            func_name<Args...>(m);                               \
        }                                                        \
                                                                 \
        pybind11::module_ m;                                     \
    }

#define ONEDAL_PY_INSTANTIATE(func_name, module, ...) \
    dal::python::instantiate<instantiator_##func_name, __VA_ARGS__>(module)

namespace oneapi::dal::python {

template <typename Index, typename T>
struct iterator {
    iterator(pybind11::module_& m) : m(m) {}

    template <typename... Args>
    void run() {
        iterate_head<Index, Args...>();
    }

    template <typename Y, typename... Args>
    void iterate_head() {
        if constexpr (is_type_list<Y>::value) {
            const auto str = type_to_str<typename Y::head>()();
            pybind11::module_ sub = m;
            if (str != "") {
                sub = m.def_submodule(str.c_str());
            }
            T(sub).template run<typename Y::head, Args...>();
            if constexpr (Y::has_last) {
                iterate_head<typename Y::last, Args...>();
            }
        }
        else {
            T(m).template run<Y, Args...>();
        }
    }

    pybind11::module_ m;
};

template <typename Iter>
void instantiate_impl(pybind11::module_& m) {
    Iter(m).run();
}

template <typename Iter, typename T, typename... Args>
void instantiate_impl(pybind11::module_& m) {
    instantiate_impl<iterator<T, Iter>, Args...>(m);
}

template <typename Instantiator, typename T, typename... Args>
void instantiate(pybind11::module_& m) {
    instantiate_impl<iterator<T, Instantiator>, Args...>(m);
}

} // namespace oneapi::dal::python
