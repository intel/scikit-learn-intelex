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

#include <array>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/interop/utils/common.hpp"

#include "onedal/interop/sua/sua_utils.hpp"
#include "onedal/interop/sua/sua_interface.hpp"
#include "onedal/interop/sua/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::sua {

template <std::size_t dim>
inline auto get_strides(const py::dict& sua, const std::array<std::int64_t, dim>& shape) {
    const auto raw_strides = sua["strides"];
    if (raw_strides.is_none()) {
        return utils::get_c_strides(shape);
    }
    else {
        auto t = raw_strides.cast<py::tuple>();
        return utils::convert_tuple<dim>(t);
    }
}

template <std::int64_t dim>
inline auto get_shape(const py::dict& sua) {
    auto t = sua["shape"].cast<py::tuple>();
    return utils::convert_tuple<dim>(t);
}

template <std::int64_t dim>
inline auto get_strides(const py::dict& sua) {
    const auto shape = get_shape<dim>(sua);
    return get_strides<dim>(sua, shape);
}

template <std::int64_t dim>
sua_interface<dim> get_sua_interface(const py::dict& sua) {
    sua_interface<dim> result;

    result.dtype = get_sua_dtype(sua);
    result.offset = std::int64_t{ 0l };
    result.data.first = get_sua_ptr(sua);
    result.data.second = is_sua_readonly(sua);
    result.shape = get_shape<dim>(sua);
    result.strides = get_strides<dim>(sua);
    result.queue = get_sua_shared_queue(sua);

    return result;
}

template <std::int64_t dim>
py::dict get_sua_interface(const sua_interface<dim>& sua) {
    py::dict result;

    result["offset"] = std::int64_t{ 0l };
    result["version"] = std::int64_t{ 1l };
    result["syclobj"] = pack_queue(sua.queue);
    result["shape"] = utils::convert_array(sua.shape);
    result["strides"] = utils::convert_array(sua.strides);
    result["typestr"] = convert_dal_to_sua_type(sua.dtype);
    result["data"] = py::make_tuple(sua.data.first, sua.data.second);

    return result;
}

#define INSTANTIATE_DIM(DIM)                                             \
    template sua_interface<DIM> get_sua_interface<DIM>(const py::dict&); \
    template py::dict get_sua_interface<DIM>(const sua_interface<DIM>&);

INSTANTIATE_DIM(1)
INSTANTIATE_DIM(2)

} // namespace oneapi::dal::python::interop::sua
