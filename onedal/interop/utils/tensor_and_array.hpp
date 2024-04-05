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

#include "onedal/common/dtype_dispatcher.hpp"
#include "onedal/interop/utils/common.hpp"
#include "onedal/interop/utils/tensor_and_array.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::utils {

template <typename Type, typename Tensor>
inline auto get_data_struct(const Tensor& tensor) {
    const bool is_readonly = tensor.data.second;
    const std::uintptr_t raw_ptr = tensor.data.first;
    const auto* const ptr = reinterpret_cast<const Type*>(raw_ptr);
    return std::make_pair(ptr, is_readonly);
}

template <typename Type, typename Tensor, typename Deleter>
inline dal::array<Type> wrap_to_array_host(const Tensor& tensor, std::int64_t count, Deleter&& del) {
    const auto [ptr, is_readonly] = get_data_struct<Type>(tensor);
    
    if (is_readonly) {
        return dal::array<Type>(ptr, //
            count, std::forward<Deleter>(del));
    }
    else {
        auto* const mut_ptr = const_cast<Type*>(ptr);
        return dal::array<Type>(mut_ptr, // 
            count, std::forward<Deleter>(del));
    }
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Type, typename Tensor, typename Deleter>
inline dal::array<Type> wrap_to_array_device(const Tensor& tensor, std::int64_t count, Deleter&& del) {
    const auto [ptr, is_readonly] = get_data_struct<Type>(tensor);
    
    if (is_readonly) {
        return dal::array<Type>(*tensor.queue, ptr, //
            count, std::forward<Deleter>(del));
    }
    else {
        auto* const mut_ptr = const_cast<Type*>(ptr);
        return dal::array<Type>(*tensor.queue, mut_ptr, // 
            count, std::forward<Deleter>(del));
    }
}
#endif // ONEDAL_DATA_PARALLEL

template <typename Type, typename Tensor, typename Deleter>
inline dal::array<Type> wrap_to_array_any(const Tensor& tensor, std::int64_t count, Deleter&& del) {
    if (tensor.queue.get() == nullptr) {
        return wrap_to_array_host<Type>(tensor, count, std::forward<Deleter>(del));
    }
#ifdef ONEDAL_DATA_PARALLEL
    else if (tensor.queue.get() != nullptr) {
        return wrap_to_array_device<Type>(tensor, count, std::forward<Deleter>(del));
    }
#endif // ONEDAL_DATA_PARALLEL
    else {
        throw std::runtime_error("Unsupported device");
    }
}

template <typename Type, typename Tensor, typename Deleter>
inline dal::array<Type> wrap_to_array_impl(const Tensor& tensor, Deleter&& del) {
    const auto [count] = tensor.shape;
    return wrap_to_array_any<Type>(tensor, count, std::forward<Deleter>(del));
}

template <typename Tensor, typename Deleter>
inline py::object wrap_to_array(const Tensor& tensor, Deleter&& del) {
    return detail::dispatch_by_data_type(tensor.dtype, 
                    [&](auto type_tag) -> py::object {
        using type_t = std::decay_t<decltype(type_tag)>;
        auto table =  wrap_to_array_impl<type_t>( //
                    std::move(tensor), std::move(del));
        return py::cast( std::move(table) );
    });
}

template <typename Type>
inline dal::data_type get_data_type(const dal::array<Type>& table) {
    return detail::make_data_type<Type>();
}

template <typename Type>
inline auto get_array_queue(const dal::array<Type>& array) {
#ifdef ONEDAL_DATA_PARALLEL
    if (auto queue = array.get_queue()) {
        const auto& raw = queue.value();
        return std::make_shared<sycl::queue>(raw);
    }
#endif // ONEDAL_DATA_PARALLEL
    return std::shared_ptr<sycl::queue>{};
}

template <typename Index, typename Type>
inline auto get_shape(const dal::array<Type>& array) {
    const std::int64_t raw_count = array.get_count();
    const auto count = detail::integral_cast<Index>(raw_count);

    return std::array<Index, 1ul>{ count };
}

template <typename Tensor, typename Type>
inline Tensor& fill_shape(Tensor& tensor, const dal::array<Type>& array) {
    using shape_array_t = std::decay_t<decltype(tensor.shape)>;
    using shape_t = typename shape_array_t::value_type;

    tensor.shape = get_shape<shape_t>(array);

    return tensor;
}

template <typename Tensor, typename Type>
inline Tensor& fill_strides(Tensor& tensor, const dal::array<Type>& array) {
    using strides_array_t = std::decay_t<decltype(tensor.strides)>;
    using strides_t = typename strides_array_t::value_type;

    tensor.strides = std::array<strides_t, 1ul>{ strides_t(1) };

    return tensor;
}

template <typename ArrType, typename Type = ArrType>
inline auto get_raw_data(const dal::array<ArrType>& array) {
    const ArrType* const data = array.get_data();
    const bool is_readonly = !array.has_mutable_data();
    auto raw_data = reinterpret_cast<std::uintptr_t>(data);
    return std::make_pair(raw_data, is_readonly);
}

template <typename Tensor, typename Type>
inline Tensor wrap_from_array(const dal::array<Type>& array) {
    Tensor result;

    // Fixed for now
    result.offset = 0;

    fill_shape(result, array);
    fill_strides(result, array);

    result.queue = get_array_queue(array);
    result.dtype = get_data_type(array);
    result.data = get_raw_data(array);

    return result;
}

} // namespace oneapi::dal::python::interop::utils
