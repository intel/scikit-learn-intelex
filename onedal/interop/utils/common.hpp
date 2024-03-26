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

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#else // ONEDAL_DATA_PARALLEL
namespace sycl {
    class queue;
} // namespace sycl
#endif // ONEDAL_DATA_PARALLEL

#include <array>
#include <iterator>
#include <stdexcept>

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::interop::utils {

template <typename Type, std::size_t dim>
inline auto get_c_strides(const std::array<Type, dim>& shape) -> std::array<Type, dim>{
    std::size_t stride = 1ul;
    std::array<Type, dim> result;
    for (std::size_t d = dim; 0ul < d; --d) {
        result.at(d - 1ul) = stride;
        stride *= shape.at(d - 1ul);
    }
    return result;
}

template <std::size_t dim, typename Type = std::int64_t>
inline std::array<Type, dim> convert_tuple(const py::tuple& t) {
    constexpr const char confusing_lengths[] = "Confusing lengths";
    if (t.size() != detail::integral_cast<py::ssize_t>(dim)) {
        throw std::length_error(confusing_lengths);
    }
    std::size_t i = 0ul;
    std::array<Type, dim> result;
    for (auto it = t.begin(); it != t.end(); ++it, ++i) {
        const py::ssize_t val = it->cast<py::ssize_t>();
        result.at(i) = detail::integral_cast<Type>(val);
    }
    if (t.size() != detail::integral_cast<py::ssize_t>(i)) {
        throw std::length_error(confusing_lengths);
    }
    return result;
}

template <typename Type, std::size_t dim, std::size_t... ids>
inline py::tuple convert_array_impl(const std::array<Type, dim>& t, 
                const std::index_sequence<ids...>* const = nullptr) {
    return py::make_tuple( t.at(ids)... );
}

template <typename Type, std::size_t dim>
inline py::tuple convert_array(const std::array<Type, dim>& t) {
    using indices_t = std::make_index_sequence<dim>;
    constexpr const indices_t* dummy = nullptr;
    return convert_array_impl(t, dummy);
}

template <typename Iter>
inline py::tuple to_tuple(Iter first, Iter last) {
    const auto size = std::distance(first, last);

    py::tuple result(size);
    py::ssize_t out_it = 0ul;

    for (auto inp_it = first; inp_it != last; ++inp_it) {
        result[out_it++] = py::cast( *inp_it );
    }

    return result;
}

template <typename Container>
inline py::tuple to_tuple(const Container& container) {
    const auto first = std::cbegin(container);
    const auto last = std::cend(container);

    return to_tuple(first, last);
}

} // namespace oneapi::dal::python::interop::utils
