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

#include "onedal/common.hpp"

#include "onedal/interop/interop.hpp"

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(interop) {
    using namespace interop;
    auto sub_module = m.def_submodule("interop");
    instantiate_sua_interop(sub_module);
    instantiate_buffer_interop(sub_module);
    instantiate_dlpack_interop(sub_module);
}

} // namespace oneapi::dal::python