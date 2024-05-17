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
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_iface.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"

#include "onedal/common.hpp"
#include "onedal/common/dtype_dispatcher.hpp"

#include "onedal/interop/common.hpp"
#include "onedal/interop/sua/sua_utils.hpp"
#include "onedal/interop/sua/sua_interface.hpp"
#include "onedal/interop/sua/sua_and_array.hpp"
#include "onedal/interop/sua/sua_and_table.hpp"
#include "onedal/interop/sua/dtype_conversion.hpp"
#include "onedal/interop/utils/tensor_and_table.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::sua {

using interface_t = sua_interface<2l>;

#ifdef ONEDAL_DATA_PARALLEL
py::object wrap_from_homogen_table(const dal::homogen_table& table) {
    using tensor_t = sua_interface<2l>;
    auto sua = utils::wrap_from_homogen_table<tensor_t>(table);
    return get_sua_interface(sua);
}

#else // ONEDAL_DATA_PARALLEL

py::object wrap_from_homogen_table(const dal::homogen_table& table) {
    throw std::runtime_error("Not able to wrap to SUA");
    return py::none();
}

#endif // ONEDAL_DATA_PARALLEL

#ifdef ONEDAL_DATA_PARALLEL

template <typename Deleter>
py::object wrap_to_homogen_table(interface_t sua, Deleter&& del) {
    return utils::wrap_to_homogen_table(sua, std::forward<Deleter>(del));
}

#else // ONEDAL_DATA_PARALLEL

template <typename Deleter>
py::object wrap_to_homogen_table(interface_t sua, Deleter&& del) {
    throw std::runtime_error("SYCL runtime is required");
    return py::none();
}

#endif // ONEDAL_DATA_PARALLEL

py::object wrap_object_to_homogen_table(py::object object) {
    const py::dict dict = get_sua_interface(object);
    const interface_t sua = get_sua_interface<2l>(dict);
    const auto deleter = [object](auto* ptr) -> void {
        //check_correctness(raw_ptr, ptr);
        object.dec_ref();
    };
    auto result = wrap_to_homogen_table( //
        std::move(sua),
        std::move(deleter));
    object.inc_ref();
    return result;
}

py::object wrap_dict_to_homogen_table(py::dict dict) {
    auto sua = get_sua_interface<2l>(dict);
    auto deleter = [](auto* ptr) -> void {
        //check_correctness(raw_ptr, ptr);
    };
    return wrap_to_homogen_table(std::move(sua), std::move(deleter));
}

void instantiate_wrap_to_homogen_table(py::module& pm) {
    constexpr const char name[] = "wrap_to_homogen_table";
    pm.def(name, &wrap_object_to_homogen_table);
    pm.def(name, &wrap_dict_to_homogen_table);
}

void instantiate_wrap_from_homogen_table(py::module& pm) {
    constexpr const char name[] = "wrap_from_homogen_table";
    pm.def(name, &wrap_from_homogen_table);
}

void instantiate_sua_and_homogen_table(py::module& pm) {
    instantiate_wrap_from_homogen_table(pm);
    instantiate_wrap_to_homogen_table(pm);
}

void instantiate_sua_and_table(py::module& pm) {
    instantiate_sua_and_homogen_table(pm);
}

} // namespace oneapi::dal::python::interop::sua
