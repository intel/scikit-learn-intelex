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

#define ONEDAL_PY_INIT_MODULE(name) void init_##name(pybind11::module_& m)

#define DEF_ONEDAL_PY_PROPERTY(name, parent) \
    def_property(#name, &parent::get_##name, &parent::set_##name)

#define DEF_ONEDAL_PY_PROPERTY_T(name, parent, T) \
    def_property(#name, &parent::template get_##name<T>, &parent::template set_##name<T>)
