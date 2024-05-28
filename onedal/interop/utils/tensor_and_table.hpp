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

#pragma once

#include <pybind11/pybind11.h>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_iface.hpp"
#include "oneapi/dal/table/detail/table_utils.hpp"

#include "onedal/common/dtype_dispatcher.hpp"
#include "onedal/interop/utils/common.hpp"
#include "onedal/interop/utils/tensor_and_array.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::utils {

template <typename Tensor>
inline dal::data_layout get_layout(const Tensor& tensor) {
    const auto& [rc, cc] = tensor.shape;
    const auto& [rs, cs] = tensor.strides;
    using shape_t = std::decay_t<decltype(rc)>;
    using stride_t = std::decay_t<decltype(rs)>;
    constexpr auto one = static_cast<shape_t>(1);
    static_assert(std::is_same_v<shape_t, stride_t>);

    if (rs == cc && cs == one) {
        return dal::data_layout::row_major;
    }
    else if (rs == one && cs == rc) {
        return dal::data_layout::column_major;
    }
    else {
        throw std::runtime_error("Wrong strides");
    }
}

template <typename Type, typename Tensor, typename Deleter>
inline dal::homogen_table wrap_to_homogen_table_impl(const Tensor& tensor, Deleter&& del) {
    const auto& [raw_rc, raw_cc] = tensor.shape;
    const dal::data_layout layout = get_layout(tensor);
    const auto rc = detail::integral_cast<std::int64_t>(raw_rc);
    const auto cc = detail::integral_cast<std::int64_t>(raw_cc);
    const auto count = detail::check_mul_overflow<std::int64_t>(rc, cc);

    auto array = wrap_to_array_any<Type>(tensor, count, std::forward<Deleter>(del));

    return dal::homogen_table::wrap(std::move(array), rc, cc, layout);
}

template <typename Tensor, typename Deleter>
inline py::object wrap_to_homogen_table(const Tensor& tensor, Deleter&& del) {
    return detail::dispatch_by_data_type(tensor.dtype, 
                    [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        auto table =  wrap_to_homogen_table_impl<type_t>( //
                            std::move(tensor), std::move(del));
        return py::cast( std::move(table) );
    });
}

template <typename Type>
inline auto get_shape(const dal::homogen_table& table) {
    const std::int64_t raw_row_count = table.get_row_count();
    const std::int64_t raw_column_count = table.get_column_count();

    const auto row_count = detail::integral_cast<Type>(raw_row_count);
    const auto column_count = detail::integral_cast<Type>(raw_column_count);

    return std::array<Type, 2ul>{ row_count, column_count };
}

template <typename Tensor>
inline Tensor& fill_shape(Tensor& tensor, const dal::homogen_table& table) {
    using shape_array_t = std::decay_t<decltype(tensor.shape)>;
    using shape_t = typename shape_array_t::value_type;

    tensor.shape = get_shape<shape_t>(table);

    return tensor;
}

template <typename Tensor>
inline Tensor& fill_strides(Tensor& tensor, const dal::homogen_table& table) {
    using strides_array_t = std::decay_t<decltype(tensor.strides)>;
    using strides_t = typename strides_array_t::value_type;

    const auto [row_count, column_count] = get_shape<strides_t>(table);

    const auto layout = table.get_data_layout();
    if (layout == dal::data_layout::column_major) {
        tensor.strides = strides_array_t{ //
                    strides_t(1), row_count };
    }
    else if (layout == dal::data_layout::row_major) {
        tensor.strides = strides_array_t{ //
                    column_count, strides_t(1) };
    }
    else {
        throw std::runtime_error("Unknown data layout");
    }

    return tensor;
}

inline dal::data_type get_data_type(const homogen_table& table) {
    return table.get_metadata().get_data_type(0l);
}

inline auto get_data_array(const dal::homogen_table& table) {
    auto iface = detail::get_homogen_table_iface(table);
    return dal::array<dal::byte_t>{ iface->get_data() };
}

template <typename Tensor>
inline Tensor wrap_from_homogen_table(const dal::homogen_table& table) {
    Tensor result;

    // Fixed for now
    result.offset = 0;

    fill_shape(result, table);
    fill_strides(result, table);

    auto array = get_data_array(table);
    result.queue = get_array_queue(array);
    result.dtype = get_data_type(table);
    result.data = get_raw_data(array);

    return result;
}

} // namespace oneapi::dal::python::interop::utils
