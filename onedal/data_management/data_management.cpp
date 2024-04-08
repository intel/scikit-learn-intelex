/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "onedal/data_management/array/array.hpp"
#include "onedal/data_management/table/table.hpp"
#include "onedal/data_management/csr_table/csr_table.hpp"
#include "onedal/data_management/homogen_table/homogen_table.hpp"
#include "onedal/data_management/table_metadata/table_metadata.hpp"

#include "onedal/data_management/data_management.hpp"

namespace oneapi::dal::python {

ONEDAL_PY_INIT_MODULE(data_management) {
    auto sub_module = m.def_submodule("data_management");

    data_management::instantiate_array(sub_module);

    data_management::instantiate_table(sub_module);
    data_management::instantiate_csr_table(sub_module);
    data_management::instantiate_table_enums(sub_module);
    data_management::instantiate_homogen_table(sub_module);
    data_management::instantiate_table_metadata(sub_module);
}

} // namespace oneapi::dal::python
